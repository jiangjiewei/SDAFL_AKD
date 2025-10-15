import torch.nn.functional as F
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Classification(object):
    def __init__(self):
        self.init()
    
    
    def init(self):
        self.pred_list = [] # 用于存储预测结果的列表
        self.label_list = [] # 用于存储标签的列表
        self.correct_count = 0 # 正确预测的数量
        self.total_count = 0 # 总预测的数量
        self.loss = 0  # 总损失值
    
    def update(self, pred, label, easy_model=False):  # 更新方法，接收预测结果pred、标签label和简易模型标志easy_model
        pred = pred.cpu()  # 将预测结果移动到cpu
        label = label.cpu()# 将移动标签移动到cpu
        
        if easy_model: # 如果是简易模型，则不计算损失
            pass
        else:  # 否则计算交叉熵损失并累加到总损失
            # 假如是单个的话 那不就是单纯的损失值吗？
            class_weights = torch.FloatTensor([1.5, 1.0, 3.0])
            loss = F.cross_entropy(pred, label, weight=class_weights).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]  # 获取预测最大概率的索引作为预测类别
        self.pred_list.extend(pred.numpy()) # 将预测结果添加到列表
        self.label_list.extend(label.numpy()) # 将标签添加到列表
        # 计算正确预测的数量，pred.eq(label.data.view_as(pred))生成一个由布尔值组成的张量，表示预测是否正确
        # 这个表达式是在进行元素级别的比较，检查pred张量中预测的类别是否与label张量中的实际类别相等。
        # 即统计预测正确的个数
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label) # 更新总预测数量
            
    def results(self): # 结果获取方法
        result_dict = {}  # 创建一个字典用于存储结果
        # 计算并存储准确率，使用self.correct_count和self.total_count
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        # 计算并存储平均损失，使用self.loss和self.total_count
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        # 重置统计指标为初始状态
        self.init()
        # 返回结果字典
        return result_dict
