import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_block(nn.Module):
    def __init__(self, inchannels, reduction = 16 ):
        super(SE_block,self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1,1,1))
        self.FC1 = nn.Linear(inchannels,inchannels//reduction)
        self.FC2 = nn.Linear(inchannels//reduction,inchannels)

    def forward(self,x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0),x.size(1),1,1,1)
        return model_input * x

class AC_layer(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(AC_layer,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,3,3),stride=1,padding=1,bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,1,3),stride=1,padding=(0,0,1),bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels,outchannels,(1,3,1),stride=1,padding=(0,1,0),bias=False),
            nn.BatchNorm3d(outchannels))
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4

class dense_layer(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(dense_layer,self).__init__()
        self.block = nn.Sequential(
            AC_layer(inchannels,outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            AC_layer(outchannels,outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            SE_block(outchannels),
            nn.MaxPool3d(2,2),
        )

    def forward(self,x):
        new_features = self.block(x)
        x = F.max_pool3d(x,2)
        x = torch.cat([x, new_features], 1)

        return x

class ScaleDense(nn.Module):
    def __init__(self,nb_filter=8, nb_block=5, use_gender=True):
        '''
        Develop Scale Dense for brain age estimation

        Args:
            nb_filter (int): number of initial convolutional layer filter. Default: 8
            nb_block (int): number of Dense block. Default: 5
            use_gender (bool, optional): if use gender input. Default: True
        '''
        super(ScaleDense,self).__init__()
        self.nb_block = nb_block
        self.use_gender = use_gender
        self.pre = nn.Sequential(
            nn.Conv3d(1,nb_filter,kernel_size=7,stride=1
                     ,padding=1,dilation=2),
            nn.ELU(),
            )
        self.block, last_channels = self._make_block(nb_filter,nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.deep_fc = nn.Sequential(
            nn.Linear(last_channels,32,bias=True),
            nn.ELU(),
            )

        self.male_fc = nn.Sequential(
            nn.Linear(2,16,bias=True),
            nn.Linear(16,8,bias=True),
            nn.ELU(),
            )
        self.end_fc_with_gender = nn.Sequential(
            nn.Linear(40,16),
            nn.Linear(16,1),
            nn.ReLU()
            )
        self.end_fc_without_gender = nn.Sequential(
            nn.Linear(32,16),
            nn.Linear(16,1),
            nn.ReLU()
            )


    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer(inchannels,outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x, male_input):
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.deep_fc(x)
        if self.use_gender:
            male = torch.reshape(male_input,(male_input.size(0),-1))
            male = self.male_fc(male)
            x = torch.cat([x,male.type_as(x)],1)
            x = self.end_fc_with_gender(x)
        else:
            x = self.end_fc_without_gender(x)
        return x

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num /1e6}

