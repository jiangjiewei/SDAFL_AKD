import os
from network.get_network import GetNetwork
import torch
from typing import Dict, List, OrderedDict
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 打乱数据集的函数，以支持随机化训练
# 此处更新全局和局部？


def compute_mc_loss(outputs, global_outputs, temperature=1):
    cos_sim = nn.CosineSimilarity(dim=1)
    pos_sim = cos_sim(outputs, global_outputs)
    # 仅保留正样本相似度，简化损失计算
    mc_loss = -torch.mean(pos_sim)
    #correction_direction = pos_sim.mean()

    return mc_loss


def epoch_site_train(epochs, site_name, model, optimzier, scheduler, dataloader, log_ten, metric, global_model):
    model.train()  # 设置模型为训练模式
    for i, data_list in enumerate(dataloader):  # 遍历数据加载器
        imgs, labels, domain_labels = data_list  # 解压数据
        imgs = imgs.cuda()
        labels = labels.cuda()
        optimzier.zero_grad()
        output = model(imgs)
        # 这里是求均值的损失
        ce_loss = F.cross_entropy(output, labels)  # 这里可以修改注
        global_outputs = global_model(imgs)
        mc_loss = compute_mc_loss(output, global_outputs)
        loss = ce_loss + mc_loss
        loss.backward()
        optimzier.step()
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs * len(dataloader) + i)  # 记录训练损失
        # 这里是均值损失*数量还会叠加
        metric.update(output, labels)

    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    # 更新学习率
    scheduler.step()

    # 每个


def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, global_model):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        epoch_site_train(comm_rounds * args.local_epochs + local_epoch, site_name, model, optimizer, scheduler,
                         dataloader, log_ten, metric, global_model)


def site_evaluation(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(
        f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs * epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"] * 100:.2f}%')

    return results_dict


# 这个没用因为没写class_acc这个方法
def site_evaluation_class_level(epochs, site_name, args, model, dataloader, log_file, log_ten, metric,
                                note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_class_acc', results_dict['class_level_acc'], epochs)
    log_file.info(
        f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs * epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"] * 100:.2f}% | C Acc: {results_dict["class_level_acc"] * 100:.2f}%')

    return results_dict


def site_only_evaluation(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict


def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}

    if args.dataset == 'pacs':
        domain_list = pacs_domain_list
    elif args.dataset == 'officehome':
        domain_list = officehome_domain_list
    elif args.dataset == 'domainNet':
        domain_list = domainNet_domain_list
    elif args.dataset == 'terrainc':
        domain_list = terra_incognita_list

    for domain_name in domain_list:
        model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = model_dict[domain_name].cuda()
        optimizer_dict[domain_name] = torch.optim.SGD(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4)
        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name],
                                                                          step_size=int(total_epochs * 0.8), gamma=0.1)
        elif args.lr_policy == 'mul_step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(optimizer_dict[domain_name],
                                                                               milestones=[int(total_epochs * 0.3),
                                                                                           int(total_epochs * 0.8)],
                                                                               gamma=0.1)
        elif args.lr_policy == 'exp95':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name],
                                                                                 gamma=0.95)
        elif args.lr_policy == 'exp98':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name],
                                                                                 gamma=0.98)
        elif args.lr_policy == 'exp99':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name],
                                                                                 gamma=0.99)
        elif args.lr_policy == 'cos':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dict[domain_name],
                                                                                     T_max=total_epochs)

    return global_model, model_dict, optimizer_dict, scheduler_dict


def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    check_dict = {'args': args, 'epochs': epochs, 'model': model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(check_dict, os.path.join(path, note + '.pt'))


