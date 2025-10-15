def Dict_weight(dict_in, weight_in):  # dict_in*weight_in
    for k, v in dict_in.items():
        dict_in[k] = weight_in * v
    return dict_in


def Dict_Add(dict1, dict2):  # dict1 + dict2
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1


def Dict_Minus(dict1, dict2):  # dict1 - dict2
    for k, v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1


def Cal_Weight_Dict(dataset_dict, site_list=None):  # 函数计算所有数据集的总长度，并为每个数据集分配一个权重，该权重是数据集长度与总长度的比例
    if site_list is None:
        site_list = list(dataset_dict.keys())
    weight_dict = {}
    total_len = 0
    for site_name in site_list:
        total_len += len(dataset_dict[site_name]['test'])
    for site_name in site_list:
        site_len = len(dataset_dict[site_name]['test'])
        weight_dict[site_name] = site_len / total_len
    return weight_dict


def FedAvg(model_dict, weight_dict, global_model=None):
    new_model_dict = None
    for model_name in weight_dict.keys():
        model = model_dict[model_name]
        model_state_dict = model.state_dict()
        if new_model_dict is None:
            new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
        else:
            new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))

    if global_model is None:
        return new_model_dict
    else:
        global_model.load_state_dict(new_model_dict)
        return new_model_dict


# 刚好此处去找找动量更新的用在此处
def FedAvg_with_Momentum(model_dict, weight_dict, global_model, alpha=0.99):
    """
    使用动量更新机制的联邦平均算法。

    参数:
        model_dict: 包含所有客户端模型的字典，键为模型名称，值为模型对象。
        weight_dict: 包含每个客户端模型的权重字典，键为模型名称，值为权重。
        global_model: 全局模型对象。
        alpha: 动量系数，默认为 0.99。
    """
    new_model_dict = None
    for model_name in weight_dict.keys():
        model = model_dict[model_name]
        model_state_dict = model.state_dict()
        if new_model_dict is None:
            new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
        else:
            new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))

    if global_model is None:
        return new_model_dict
    else:
        MomentumUpdate(new_model_dict, global_model, alpha)


def FedUpdate(model_dict, global_model):
    global_model_parameters = global_model.state_dict()
    for site_name in model_dict.keys():
        model_dict[site_name].load_state_dict(global_model_parameters)
    return None




def MomentumUpdate(model, teacher, alpha=0.99):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k, v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1 - alpha) * model_dict[k]
    teacher.load_state_dict(teacher_dict)