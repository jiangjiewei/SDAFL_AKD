import os
import argparse
from utils.log_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
from data.test_dataset1 import PACS_FedDG  # pacs_dataset
from utils.classification_metric import Classification
from utils.fed_merge_moga import FedAvg, FedUpdate
from utils.trainval_func_moga import site_train, site_evaluation, GetFedModel, SaveCheckPoint
from utils.weight_adjust_moga import refine_weight_dict_by_GA
import torch


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='densenet121',
                        choices=['resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    # 每个 p a c s 中还有七种类
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=3)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=40)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--fair", type=str, default='loss', choices=['acc', 'loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='generalization_adjustment')
    parser.add_argument('--display', help='display in controller', action='store_true')

    return parser.parse_args()

def main():
    ''' log  '''
    file_name = 'moon+ga' + os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)

    '''dataset and dataloader'''
    # 初始化PACS数据集对象  此处定义测试集除外的就是训练集 好像是把除了4个域每个域都有训练测试和验证
    dataobj = PACS_FedDG(batch_size=args.batch_size)  # test_domain=args.test_domain,
    # 获取数据加载器和数据集字典
    dataloader_dict, dataset_dict = dataobj.GetData()

    '''model'''
    # 模型、优化器、学习率调度器等初始化
    metric = Classification()  # 初始化分类指标
    # 获取联邦学习模型和优化器等
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = {}  # 初始化权重字典
    site_results_before_avg = {}  # 初始化本地训练前的结果字典
    site_results_after_avg = {}  # 初始化本地训练后的结果字典
    #l_correction_direction = {}  # 初始化纠正方向
    #g_correction_direction = {}  # 初始化纠正方向


    # 初始化每个训练站点的权重
    for site_name in dataobj.train_domain_list:
        weight_dict[site_name] = 1. / 3.  # 每个站点初始权重相同
        site_results_before_avg[site_name] = None  # 本地训练前结果初始化为None
        site_results_after_avg[site_name] = None  # 本地训练后结果初始化为None
        #l_correction_direction[site_name] = None  # 初始化纠正方向
        #g_correction_direction[site_name] = None  # 初始化纠正方向

    # 联邦学习权重更新
    FedUpdate(model_dict, global_model)  # 更新本地模型为全局模型
    # 训练和验证循环
    best_val = 0.  # 初始化最佳验证准确率
    step_size_decay = args.step_size / args.comm  # 学习率衰减的步长
    # 进行args.comm轮的联邦平均算法
    # 每个客户端（站点）进行本地训练 好像也就4个域
    for i in range(args.comm + 1):
        for domain_name in dataobj.train_domain_list:
            # 站点训练
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name],
                       scheduler_dict[domain_name], dataloader_dict[domain_name]['train'], log_ten, metric, global_model)
            # note 只是log用来区别状态的
            site_results_before_avg[domain_name] = site_evaluation(i, domain_name, args, model_dict[domain_name],
                                                                   dataloader_dict[domain_name]['test'], log_file,
                                                                   log_ten, metric, note='before_fed')
        # 执行联邦平均算法更新全局模型
        # 虽然没有传回参数但是已通过.load_state_dict更新了字典里的参数
        FedAvg(model_dict, weight_dict, global_model)
        # 再次更新本地模型为全局模型
        FedUpdate(model_dict, global_model)
        # 评估全局模型在各个站点的验证集上的表现
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(i, domain_name, args, model_dict[domain_name],
                                                                  dataloader_dict[domain_name]['test'], log_file,
                                                                  log_ten, metric)
            fed_val += site_results_after_avg[domain_name]['acc'] * 1/3

        if fed_val >= best_val:
            best_val = fed_val
            # 保存检查点
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            # 在测试领域上评估全局模型的表现
            log_file.info(f'Model saved! Best Val Acc: {best_val * 100:.2f}%')
        # 更新权重字典，使用GA算法优化权重
        weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg,
                                               args.step_size - (i - 1) * step_size_decay, fair_metric=args.fair)
        log_str = f'Round {i} FedAvg weight: {weight_dict}'
        log_file.info(log_str)
    # 保存最终模型检查点
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')


if __name__ == '__main__':
    main()