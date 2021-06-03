import torch
import numpy as np 
from model.sodeep import SpearmanLoss, load_sorter

'''
Ranking loss function for brain age estimation
'''
# ===== loss function of combine rankg loss, age difference loss adn MSE ========= #
class rank_difference_loss(torch.nn.Module):
    def __init__(self, sorter_checkpoint_path, beta=1):
        '''
        ['Ranking loss', which including Sprear man's ranking loss and age difference loss]

        Args:
            bate (float, optional): 
            [used as a weighte between ranking loss and age difference loss. 
            Since ranking loss is in (0,1),but age difference is relative large. 
            In order to banlance these two loss functions, beta is set in (0,1)]. 
            Defaults to 1.
        '''
        super(rank_difference_loss,self).__init__()
        self.spearman_loss = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()
        self.beta = beta

    def forward(self, mem_pred, mem_gt):
        ranking_loss = self.spearman_loss(mem_pred, mem_gt)
        a = np.random.randint(0,mem_pred.size(0),mem_pred.size(0))
        b = np.random.randint(0,mem_gt.size(0),mem_gt.size(0))
        diff_mem_pred = (mem_pred[a]-mem_pred[b])
        diff_mem_gt = (mem_gt[a]-mem_gt[b])
        age_difference_loss = torch.mean((diff_mem_pred-diff_mem_gt)**2)
        
        loss = (ranking_loss) + self.beta * age_difference_loss
        
        return loss

     
