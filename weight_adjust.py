import numpy as np


def refine_weight_dict_by_GA(weight_dict, site_before_results_dict, site_after_results_dict, step_size=0.1,
                             fair_metric='loss'):

    loss_value_list = []
    acc_value_list=[]
    for site_name in site_before_results_dict.keys():
        # 准确-损失
        # 这里是否存在0 或者nan 大概率是存在0
        # site_name 是哪个域fair_metric是准确率还是损失
        # 问题在于这个 site_after_results_dict 是个什么 是 site_evaluation的返回
        gap_loss = site_after_results_dict[site_name]['loss'] - site_before_results_dict[site_name]['loss']
        gap_acc = site_after_results_dict[site_name]['acc'] - site_before_results_dict[site_name]['acc']
        # 防止分母为0的情况
        if gap_loss == 0:
            loss_value_list.append(0.001)
        else:
            loss_value_list.append(gap_loss)
        if gap_acc == 0:
            acc_value_list.append(0.001)
        else:
            acc_value_list.append(gap_acc)

    loss_value_list =np.array(loss_value_list)
    acc_value_list=np.array(acc_value_list)
    step_size = 1. / 3. * step_size

    # 很奇怪明明聚合了效果却一样 这种问题该怎么解决z
    # 归一化
    norm_gap_list=[x - y for x, y in zip(loss_value_list, acc_value_list)]
    norm_gap_list = norm_gap_list / np.max(np.abs(norm_gap_list))
    weight_dict_temp = {}
    #

    for i, site_name in enumerate(weight_dict.keys()):
        weight_dict_temp[site_name] = weight_dict[site_name] + norm_gap_list[i] * step_size
        weight_dict[site_name] += weight_dict_temp[site_name]

    weight_dict = weight_clip(weight_dict)
    return weight_dict


def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]

    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight

    return weight_dict

