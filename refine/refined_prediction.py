import os
import torch
import CNN
import ResNet
import DenseNet
import shutil
import refined_CNN
import refined_ResNet
import refined_DenseNet
import numpy as np
import torch.nn as nn
from config import opt
import matplotlib.pyplot as plt
from load_data import IMG_Folder
from sklearn.metrics import roc_auc_score,mean_absolute_error,mean_squared_error



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    test_data = IMG_Folder(opt.excel_path, opt.test_folder)
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')

    # --------  build and set model  --------  
    if opt.model == 'CNN':
        model = refined_CNN.CNN(8,8,5)
    elif opt.model == 'resnet':
        model = refined_ResNet.resnet18()
    elif opt.model == 'DenseNet':
        model = refined_DenseNet.dense_net(8,8,5)


    model = nn.DataParallel(model).to(device)
    criterion = nn.MSELoss().to(device)

    # -------define discriminate network -------
    model_dis = CNN.CNN(8,8,5)
    model_dis = nn.DataParallel(model_dis).to(device)
    model_dis.load_state_dict(torch.load('./model//CNN_best_model.pth.tar')['state_dict'],False)
    model_dis.eval()

    # ----- build data loader --------
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=opt.batch_size, 
                                                  num_workers=opt.num_workers, pin_memory=True,
                                                  drop_last=False)
    model.load_state_dict(torch.load(os.path.join(opt.output_dir+opt.model_name))['state_dict'])


    # ---- test preformance ----------
    test(valid_loader=test_loader, model=model,model_dis=model_dis, criterion=criterion, device=device)


def test(valid_loader, model, model_dis,criterion, device,figure=False,figure_name='True_age_and_predicted_age.png'):
    losses = AverageMeter()
    MAE = AverageMeter()

    # switch to evaluate mode
    model.eval()
    out = []
    targ = []

    with torch.no_grad():
        for _, (input, id,target,male) in enumerate(valid_loader):
            input = input.type(torch.FloatTensor).to(device)
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.type(torch.FloatTensor).to(device)

            # one hot encode male lable
            male = torch.unsqueeze(male,1)
            male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
            male = male.to(device).type(torch.FloatTensor)

            dis_age = model_dis(input, male).detach()
            dis_age = discriminate_age(dis_age,range=opt.dis_range).to(device)

            # compute residual age
            residual_age = target - dis_age

            # compute output and loss
            predicted_residual_age = model(input, male, dis_age)
            output_age = predicted_residual_age + dis_age
            out.append(output_age.cpu().numpy())
            targ.append(target.cpu().numpy())
            loss = criterion(predicted_residual_age, residual_age)
            mae = metric(output_age.detach(), target.detach().cpu())

            # measure accuracy and record loss
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        targ = np.asarray(targ)
        out = np.asarray(out)
        target_numpy = []   # target numpy array
        predicted_numpy = []   # predicted age numpy array
        for i in targ:
            for n in i:
                target_numpy.append(n)
        target_numpy = np.array(target_numpy)
        
        for i in out:
            for n in i:
                predicted_numpy.append(n)
        predicted_numpy = np.array(predicted_numpy)

        errors = []        # error numpy array
        abs_errors = []    # absolute error numpy array
        for n in range(0, target_numpy.shape[0]):
            err = target_numpy[n] - predicted_numpy[n]
            abs_err = abs(err)
            errors.append(err)
            abs_errors.append(abs_err)

        abs_errors = np.array(abs_errors)
        errors = np.array(errors)
        target_numpy = np.squeeze(target_numpy,axis=1)
        predicted_numpy = np.squeeze(predicted_numpy,axis=1)



        print('----------------------------------------------------------------\n')
        print(
            'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f} \n'.format(
            len(valid_loader), loss=losses, MAE=MAE))
        cc = np.corrcoef(target_numpy,predicted_numpy)
        print('       CC:    ',cc)
        print('----------------------------------------------------------------')

        if figure is True:
            plt.figure()
            lx = np.arange(np.min(target_numpy),np.max(target_numpy))
            plt.plot(lx,lx,color='red',linestyle='--')
            plt.scatter(target_numpy,predicted_numpy)
            plt.xlabel('Age')
            plt.ylabel('predicted brain age')
            plt.savefig(opt.output_dir+figure_name)
        return MAE, cc

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)

    return mae
def discriminate_age(age, range=5):
    dis = []
    for i in age:
        value = i // range
        x = i % range
        if x < range/2:
            discri_age = value * range
        else:
            discri_age = (value+1)*range
        dis.append(discri_age)
        dis_age = np.asarray(dis,dtype='float32')
        dis_age = np.expand_dims(dis_age,axis=1)
        dis_age = torch.from_numpy(dis_age)
    return dis_age
if __name__ == "__main__":
    main()
