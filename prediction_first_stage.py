import numpy as np
import torch.nn as nn
import os,shutil,torch
import matplotlib.pyplot as plt
from utils.config import opt
from load_data import IMG_Folder
from model import ScaleDense
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error

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

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def main():
    # ======== define data loader and CUDA device ======== #
    test_data = IMG_Folder(opt.excel_path, opt.test_folder)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========  build and set model  ======== #  
    if opt.model == 'ScaleDense':
        model = ScaleDense.ScaleDense(8, 5, opt.use_gender)
    else:
        print('Wrong model choose')

    # ======== load trained parameters ======== #
    model = nn.DataParallel(model).to(device)
    criterion = nn.MSELoss().to(device)
    model.load_state_dict(torch.load(os.path.join(opt.output_dir+opt.model_name))['state_dict'])

    # ======== build data loader ======== #
    test_loader = torch.utils.data.DataLoader(test_data
                                             ,batch_size=opt.batch_size
                                             ,num_workers=opt.num_workers
                                             ,pin_memory=True
                                             ,drop_last=True
                                             )

    # ======== test preformance ======== #
    test( valid_loader=test_loader
        , model=model
        , criterion=criterion
        , device=device
        , save_npy=True
        , npy_name=opt.npz_name
        , figure=False
        , figure_name=opt.plot_name)

def test(valid_loader, model, criterion, device
        , save_npy=False,npy_name='test_result.npz'
        , figure=False, figure_name='True_age_and_predicted_age.png'):

    '''
    [Do Test process according pretrained model]

    Args:
        valid_loader (torch.dataloader): [test set dataloader defined in 'main']
        model (torch CNN model): [pre-trained CNN model, which is used for brain age estimation]
        criterion (torch loss): [loss function defined in 'main']
        device (torch device): [GPU]
        save_npy (bool, optional): [If choose to save predicted brain age in npy format]. Defaults to False.
        npy_name (str, optional): [If choose to save predicted brain age, what is the npy filename]. Defaults to 'test_result.npz'.
        figure (bool, optional): [If choose to plot and save scatter plot of predicted brain age]. Defaults to False.
        figure_name (str, optional): [If choose to save predicted brain age scatter plot, what is the png filename]. Defaults to 'True_age_and_predicted_age.png'.

    Returns:
        [float]: MAE and pearson correlation coeficent of predicted brain age in teset set.
    '''

    losses = AverageMeter()
    MAE = AverageMeter()

    model.eval() # switch to evaluate mode
    out = []
    targ = []
    ID = []
    target_numpy = []
    predicted_numpy = []
    ID_numpy = []

    print('======= start prediction =============')

    # ======= start test programmer ============= #
    with torch.no_grad():
        for _, (input, ids ,target,male) in enumerate(valid_loader):
            input = input.to(device).type(torch.FloatTensor)
            
            # ======= convert male lable to one hot type ======= #
            male = torch.unsqueeze(male,1)
            male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
            male = male.type(torch.FloatTensor).to(device)

            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.type(torch.FloatTensor).to(device)

            # ======= compute output and loss ======= #
            if opt.model == 'ScaleDense' :
                output = model(input,male)

            else:
                output = model(input)
            out.append(output.cpu().numpy())
            targ.append(target.cpu().numpy())
            ID.append(ids)
            loss = criterion(output, target)
            mae = metric(output.detach(), target.detach().cpu())

            # ======= measure accuracy and record loss ======= #
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        targ = np.asarray(targ)
        out = np.asarray(out)
        ID = np.asarray(ID)

        for idx in targ:
            for i in idx:
                target_numpy.append(i)

        for idx in out:
            for i in idx:
                predicted_numpy.append(i)
        
        for idx in ID:
            for i in idx:
                ID_numpy.append(i)

        target_numpy = np.asarray(target_numpy)
        predicted_numpy = np.asarray(predicted_numpy)
        ID_numpy = np.asarray(ID_numpy)

        errors = predicted_numpy - target_numpy
        abs_errors = np.abs(errors)
        errors = np.squeeze(errors,axis=1)
        abs_errors = np.squeeze(abs_errors,axis=1)
        target_numpy = np.squeeze(target_numpy,axis=1)
        predicted_numpy = np.squeeze(predicted_numpy,axis=1)


        # ======= output several results  ======= #
        print('===============================================================\n')
        print(
            'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f} \n'.format(
            len(valid_loader), loss=losses, MAE=MAE))

        print('STD_err = ', np.std(errors))  
        print(' CC:    ',np.corrcoef(target_numpy,predicted_numpy))
        print('PAD spear man cc',spearmanr(errors,target_numpy,axis=1))
        print('spear man cc',spearmanr(predicted_numpy,target_numpy,axis=1))
        print('mean pad:',np.mean(errors))

        print('\n =================================================================')

        if save_npy:
            savepath = os.path.join(opt.output_dir,npy_name)
            np.savez(savepath 
                    ,target=target_numpy
                    ,prediction=predicted_numpy
                    ,ID=ID_numpy)

        # ======= Draw scatter plot of predicted age against true age ======= #
        if figure is True:
            plt.figure()
            lx = np.arange(np.min(target_numpy),np.max(target_numpy))
            plt.plot(lx,lx,color='red',linestyle='--')
            plt.scatter(target_numpy,predicted_numpy)
            plt.xlabel('Chronological Age')
            plt.ylabel('predicted brain age')
            # plt.show()
            plt.savefig(opt.output_dir+figure_name)
        return MAE ,np.corrcoef(target_numpy,predicted_numpy)


if __name__ == "__main__":
    main()
