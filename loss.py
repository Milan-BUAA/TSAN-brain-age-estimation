import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt

'''
Ranking loss function for brain age estimation
'''

#  ======== ranking loss function ============= #
def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)  
    rank = torch.argsort(rank, dim=dim)         
    rank = (rank * -1) + batch_score.size(dim)  
    rank = rank.float()
    rank = rank / batch_score.size(dim)         

    return rank

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

    def __init__(self,beta=1):
        super(rank_difference,self).__init__()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()
        self.beta = beta

    def forward(self, mem_pred, mem_gt):
        a = np.random.randint(0,mem_pred.size(0),mem_pred.size(0))
        b = np.random.randint(0,mem_gt.size(0),mem_gt.size(0))

        rank_gt = get_rank(mem_gt)
        rank_pred = get_rank(mem_pred)
        ranking_loss = self.criterion_mse(rank_pred, rank_gt)

        diff_mem_pred = (mem_pred[a]-mem_pred[b])
        diff_mem_gt = (mem_gt[a]-mem_gt[b])
        age_difference_loss = torch.mean((diff_mem_pred-diff_mem_gt)**2)
        
        loss = (ranking_loss) + self.beta * age_difference_loss
        return loss

     
