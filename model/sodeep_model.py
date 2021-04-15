"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2019 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2019, June).
    SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates.
    In Proceedings of CVPR

Author: Martin Engilberge
"""


from os import stat
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
    rank = stats.rankdata(batch_score)
    rank = stats.rankdata(rank) - 1    
    rank = (rank * -1) + batch_score.size(dim)
    rank = torch.from_numpy(rank)
    rank = rank.float()
    rank = rank / batch_score.size(dim)  
    return rank

def model_loader(model_type, seq_len, pretrained_state_dict=None):

    if model_type == "lstm":
        model = lstm_baseline(seq_len)
    elif model_type == "grus":
        model = gru_sum(seq_len)
    elif model_type == "gruc":
        model = gru_constrained(seq_len)
    elif model_type == "grup":
        model = gru_proj(seq_len)
    elif model_type == "exa":
        model = sorter_exact()
    elif model_type == "lstmla":
        model = lstm_large(seq_len)
    elif model_type == "lstme":
        model = lstm_end(seq_len)
    elif model_type == "mlp":
        model = mlp(seq_len)
    elif model_type == "cnn":
        return cnn(seq_len)
    else:
        raise Exception("Model type unknown", model_type)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)

    return model


class UpdatingWrapper(nn.Module):
    """ Wrapper to store the data forwarded throught the sorter and use them later to finetune the sorter on real data
        Once enough data have been colected a call to the method update_sorter will perform the finetuning of the sorter.
    """
    def __init__(self, sorter, lr_sorter=0.00001):
        super(UpdatingWrapper, self).__init__()
        self.sorter = sorter

        self.opti = torch.optim.Adam(self.sorter.parameters(), lr=lr_sorter, betas=(0.9, 0.999))
        self.criterion = nn.L1Loss()

        self.average_loss = list()

        self.collected_data = list()

        self.nb_update = 10

    def forward(self, input_):
        out = self.sorter(input_)

        self.collected_data.append(input_.detach().cpu())
        return out

    def update_sorter(self):

        for input_opti in self.collected_data:
            self.opti.zero_grad()

            input_opti = input_opti.cuda()
            input_opti.requires_grad = True

            rank_gt = get_rank(input_opti)

            out_opti = self.sorter(input_opti)

            loss = self.criterion(out_opti, rank_gt)
            loss.backward()
            self.opti.step()

            self.average_loss.append(loss.item())

            # Empty collected data
            self.collected_data = list()

    def save_data(self, path):
        torch.save(self.collected_data, path)

    def get_loss_average(self, windows=50):
        return sum(self.average_loss[-windows:]) / min(len(self.average_loss), windows)


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


class sorter_exact(nn.Module):

    def __init__(self):
        super(sorter_exact, self).__init__()

    def comp(self, inpu):
        in_mat1 = torch.triu(inpu.repeat(inpu.size(0), 1), diagonal=1)
        in_mat2 = torch.triu(inpu.repeat(inpu.size(0), 1).t(), diagonal=1)

        comp_first = (in_mat1 - in_mat2)
        comp_second = (in_mat2 - in_mat1)

        std1 = torch.std(comp_first).item()
        std2 = torch.std(comp_second).item()

        comp_first = torch.sigmoid(comp_first * (6.8 / std1))
        comp_second = torch.sigmoid(comp_second * (6.8 / std2))

        comp_first = torch.triu(comp_first, diagonal=1)
        comp_second = torch.triu(comp_second, diagonal=1)

        return (torch.sum(comp_first, 1) + torch.sum(comp_second, 0) + 1) / inpu.size(0)

    def forward(self, input_):
        out = [self.comp(input_[d]) for d in range(input_.size(0))]
        out = torch.stack(out)

        return out.view(input_.size(0), -1)


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


if __name__ == "__main__":
    input = np.array([1,3,3,5,5,7],dtype=np.float32)
    input = torch.FloatTensor(torch.from_numpy(input))
    print(input)
    rank = get_rank(input,dim=0)
    tied_rank = get_tiedrank(input, dim=0)
    print('get rank scores:', rank)
    print('get tied rank scores:', tied_rank)