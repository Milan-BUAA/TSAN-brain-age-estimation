import torch
import torch.nn as nn
import scipy.stats.stats as stats
import numpy as np

def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)  
    rank = torch.argsort(rank, dim=dim)         
    rank = (rank * -1) + batch_score.size(dim)  
    rank = rank.float()
    rank = rank / batch_score.size(dim)         

    return rank

def get_tiedrank(batch_score, dim=0):
    batch_score = batch_score.cpu()
    rank = stats.rankdata(batch_score)
    rank = stats.rankdata(rank) - 1    
    rank = (rank * -1) + batch_score.size(dim)
    rank = torch.from_numpy(rank).cuda()
    rank = rank.float()
    rank = rank / batch_score.size(dim)  
    return rank

def model_loader(model_type, seq_len, pretrained_state_dict=None):

    if model_type == "lstm":
        model = lstm_baseline(seq_len)
    elif model_type == "lstmla":
        model = lstm_large(seq_len)
    elif model_type == "lstme":
        model = lstm_end(seq_len)
    elif model_type == "mlp":
        model = mlp(seq_len)
    else:
        raise Exception("Model type unknown", model_type)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)

    return model

class lstm_baseline(nn.Module):
    def __init__(self, seq_len):
        super(lstm_baseline, self).__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 256)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = self.conv1(out)

        return out.view(input_.size(0), -1)

class mlp(nn.Module):
    def __init__(self, seq_len):
        super(mlp, self).__init__()
        self.lin1 = nn.Linear(seq_len, 2048)
        self.lin2 = nn.Linear(2048, 2048)
        self.lin3 = nn.Linear(2048, seq_len)

        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1)
        out = self.lin1(input_)
        out = self.lin2(self.relu(out))
        out = self.lin3(self.relu(out))

        return out.view(input_.size(0), -1)

class lstm_end(nn.Module):
    def __init__(self, seq_len):
        super(lstm_end, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.GRU(self.seq_len, 5 * self.seq_len, batch_first=True, bidirectional=False)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1).repeat(1, input_.size(1), 1).view(input_.size(0), input_.size(1), -1)
        _, out = self.lstm(input_)

        out = out.view(input_.size(0), self.seq_len, -1)  # .view(input_.size(0), -1)[:,:self.seq_len]
        out = out.sum(dim=2)

        return out

class lstm_large(nn.Module):

    def __init__(self, seq_len):
        super(lstm_large, self).__init__()
        self.lstm = nn.LSTM(1, 512, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 1024)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = self.conv1(out)

        return out.view(input_.size(0), -1)


def load_sorter(checkpoint_path):
    sorter_checkpoint = torch.load(checkpoint_path)

    model_type = sorter_checkpoint["args_dict"].model_type
    seq_len = sorter_checkpoint["args_dict"].seq_len
    state_dict = sorter_checkpoint["state_dict"]

    return model_type, seq_len, state_dict



class SpearmanLoss(torch.nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set lbd to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=0):
        super(SpearmanLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()

        self.lbd = lbd

    def forward(self, mem_pred, mem_gt, pr=False):
        rank_gt = get_tiedrank(mem_gt)

        rank_pred = self.sorter(mem_pred.unsqueeze(0)).view(-1)
        return self.criterion_mse(rank_pred, rank_gt) + self.lbd * self.criterionl1(mem_pred, mem_gt)
