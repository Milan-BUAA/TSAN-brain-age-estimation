import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt

'''
Additional loss function for age prediction
include: age difference, ranking loss, PAD spearman ranking loss
'''

#  ======== AGE difference loss function ============= #
class AGE_difference(nn.Module):

    '''
    计算配对样本：真实年龄差与预测年龄差的差值，并进行均值
    '''
    def __init__(self,difference_way='power'):
        super(AGE_difference,self).__init__()
        self.difference_way = difference_way

    def forward(self, x, y):

        # ===== select pair of sample ======= #
        a = np.random.randint(0,x.size(0),x.size(0))
        b = np.random.randint(0,x.size(0),x.size(0))

        # ===== compute the difference of 
        # true age difference and predicted age difference ======== #
        diff_x = (x[a]-x[b])
        diff_y = (y[a]-y[b])
        loss = torch.nn.MSELoss(diff_x-diff_y)

        return loss


#  ======== ranking loss function ============= #
def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)  #对输入的预测年龄排序
    rank = torch.argsort(rank, dim=dim)         #对序号再进行排序
    rank = (rank * -1) + batch_score.size(dim)  #序号越靠前，rank评分越高
    rank = rank.float()
    rank = rank / batch_score.size(dim)         #对ranking得分归一化

    return rank

class SpearmanLoss(torch.nn.Module):

    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.
    """
    def __init__(self):
        super(SpearmanLoss, self).__init__()
        
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()
    def forward(self, mem_pred, mem_gt):
        rank_gt = get_rank(mem_gt)
        rank_pred = get_rank(mem_pred)
        return self.criterion_mse(rank_pred, rank_gt)


# ===== loss function of combine rankg loss, age difference loss adn MSE ========= #
class rank_difference(torch.nn.Module):
    '''
    define 'ranking loss', which including Sprear man's ranking loss and 
            age difference loss \\
    beta: is defined to be used as a weighte between ranking loss and age difference loss. Since ranking loss is in (0,1)，
    but age difference is relative large. in order to banlance these two loss functions, beta is set in (0,1)\\
    num_pair: is define to choose the number of pairs which are used to compute the age difference loss
    the total number of pair is batch size * num_pair  
    '''

    def __init__(self,beta=1,num_pair=40):
        super(rank_difference,self).__init__()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()
        self.beta = beta
        self.num_pair = num_pair

    def forward(self, mem_pred, mem_gt):
        a = np.random.randint(0,mem_pred.size(0),mem_pred.size(0)*self.num_pair)
        b = np.random.randint(0,mem_gt.size(0),mem_gt.size(0)*self.num_pair)

        rank_gt = get_rank(mem_gt)
        rank_pred = get_rank(mem_pred)
        ranking_loss = self.criterion_mse(rank_pred, rank_gt)

        diff_mem_pred = (mem_pred[a]-mem_pred[b])
        diff_mem_gt = (mem_gt[a]-mem_gt[b])
        age_difference_loss = torch.mean((diff_mem_pred-diff_mem_gt)**2)
        
        loss = (ranking_loss) + self.beta * age_difference_loss
        return loss

# ===== loss function of combine rankg loss,PAD ranking loss, age difference loss adn MSE ========= #
class PAD_rank_difference(torch.nn.Module):
    def __init__(self,lbd=1):
        super(PAD_rank_difference,self).__init__()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()
        self.lbd = lbd

    def forward(self, mem_pred, mem_gt):
        a = np.random.randint(0,mem_pred.size(0),mem_pred.size(0)*40)
        b = np.random.randint(0,mem_gt.size(0),mem_gt.size(0)*40)

        rank_gt = get_rank(mem_gt)
        rank_PAD = get_rank(mem_pred-mem_gt)
        PAD_ranking_loss = self.criterion_mse(rank_PAD, rank_gt)

        rank_gt = get_rank(mem_gt)
        rank_pred = get_rank(mem_pred)
        ranking_loss = self.criterion_mse(rank_pred, rank_gt)

        diff_mem_pred = (mem_pred[a]-mem_pred[b])
        diff_mem_gt = (mem_gt[a]-mem_gt[b])
        age_difference_loss = self.criterion_mse(diff_mem_pred,diff_mem_gt)
        
        loss = 20 * (PAD_ranking_loss+ranking_loss + age_difference_loss) + self.criterion_mse(mem_pred, mem_gt)
        return loss


# ======== use a simple neural network to test loss function =========


# input_size = 1
# output_size = 1
# num_epochs = 4000
# learning_rate = 0.001
  
# x_train = np.array([[1], [2], [3], [4], [5], [6], [7],[8]
#                     ], dtype=np.float32)

# y_train = np.array([[10], [20], [30], [40], [50], [60], [70],[80]
#                     ], dtype=np.float32)

# ========  Linear regression model  ======== 
# model = nn.Sequential(nn.Linear(input_size, 16),
#                       nn.Linear(16,1)
# )

# # Loss and optimizer 
# ========= 自定义函数 ============ #
# # criterion = AGE_difference()
# # criterion = nn.MSELoss()
# # criterion = rank_difference(lbd=5)
# criterion = PAD_rank_difference(10)

#  ======== 定义迭代优化算法， 使用的是随机梯度下降算法 =========
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
# loss_dict = []

# ========  Train the model 
# for epoch in range(num_epochs):
#     # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
#     inputs = torch.from_numpy(x_train)
#     targets = torch.from_numpy(y_train)

# ========  前向传播计算网络结构的输出结果 ========  #
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     print('pre===',outputs)
#     # print('targ==',targets)
#     # print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#  =====  打印训练信息和保存 loss ==========  #
#     loss_dict.append(loss.item())
#     if (epoch+1) % 100 == 0:
#         print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, loss.item()))
        
