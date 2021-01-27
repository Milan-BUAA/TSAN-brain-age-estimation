import os
import math
import CNN
import torch
import shutil
import ResNet
import DenseNet
import datetime
import numpy as np
import tensorboardX
import refined_CNN
import refined_ResNet
import refined_DenseNet
import torch.nn as nn
from config import opt
from mix_up import mixup
from torchsummary import summary
from refined_prediction import test
from load_data import my_DataSet,IMG_Folder,move_augmentation
from sklearn.metrics import mean_absolute_error
from loss import AGE_difference,SpearmanLoss,rank_difference

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
 # ======== main function =========== #
def main(res):
    best_metric = 100
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')

    print(opt)
    print("=========== start train the age prediction model ----------")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    trans = move_augmentation()
    train_data = IMG_Folder(opt.excel_path, opt.train_folder,transforms=trans)
    valid_data = IMG_Folder(opt.excel_path, opt.valid_folder)
    
        #--------   define data loader --------  
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                num_workers=opt.num_workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=opt.batch_size, 
                                                num_workers=opt.num_workers, 
                                                pin_memory=True,
                                                drop_last=False)

    # --------  build and set model  --------  
    if opt.model == 'CNN':
        model = refined_CNN.CNN(8,8,5)
    elif opt.model == 'resnet':
        model = refined_ResNet.resnet18()
    elif opt.model == 'DenseNet':
        model = refined_DenseNet.dense_net(8,8,5)

    model = nn.DataParallel(model).to(device)
    model_test = model

    # -------define discriminate network -------
    model_dis = CNN.CNN(8,8,5)
    model_dis = nn.DataParallel(model_dis).to(device)
    # model_dis.load_state_dict(torch.load(opt.first_stage_net)['state_dict'],False)
    model_dis.load_state_dict(torch.load(opt.first_stage_net)['state_dict'])
    model_dis.eval()

    # --------define loss function and optimizer
    # --------- define the loss function
    if opt.loss == 'MAE':
        criterion = nn.L1Loss()
    elif opt.loss == 'abs':
        criterion = AGE_difference(difference_way='abs')
    elif opt.loss == 'power':
        criterion = AGE_difference(difference_way='power')
    elif opt.loss == 'S':
        criterion = SpearmanLoss()
    elif opt.loss == 'ranking':
        criterion = rank_difference(lbd=opt.lbd)
    else:
        criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr,weight_decay=0.0005,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=1,patience=3,factor=0.5)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # ------- define tensorboardX --------
    saved_metrics, saved_epos = [], []
    num_epochs = int(opt.epochs)
    sum_writer = tensorboardX.SummaryWriter(opt.outlog_dir)
    print(" ==========> Training is getting started...")
    print(" ==========> Training takes {} epochs.".format(num_epochs))

    # -------- start train ---------------
    for epoch in range(opt.epochs):
            
        # --------  train for one epoch   --------
        train(train_loader=train_loader, model=model,dis_model=model_dis, 
              criterion=criterion,optimizer=optimizer, device=device,epoch=epoch)

        # --------  evaluate on validation set --------  
        valid_loss, valid_mae = validate(valid_loader=valid_loader, model=model,
                                        dis_model=model_dis,criterion=criterion, device=device)
        # --------  learning rate decay --------  
        scheduler.step(valid_mae)
        for param_group in optimizer.param_groups:
            print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))

        # --------  write in tensorboard --------  
        sum_writer.add_scalar('valid/1loss', valid_loss, epoch)
        sum_writer.add_scalar('valid/2acc', valid_mae, epoch)
            

        # --------  remember best metric and save checkpoint --------  
        valid_metric = valid_mae
        is_best = False
        if valid_metric < best_metric:
            is_best = True
            best_metric = min(valid_metric, best_metric)
            
            saved_metrics.append(valid_metric)
            saved_epos.append(epoch)
            print('=======>   Best at epoch %d, valid MAE %f\n' % (epoch, best_metric))
        save_checkpoint({
                        'epoch': epoch,
                        'arch': opt.model,
                        'state_dict': model.state_dict()},
                        is_best, opt.output_dir,model_name=opt.model)

        # -------- early_stopping needs the validation loss or MAE to check if it has decresed
        early_stopping(valid_mae)
        if early_stopping.early_stop:
            print("======= Early stopping =======")
            break


    # -------- write traning and validation log ----------
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epos))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    try:
        for i in range(10):
            print('{:03} {:.3f}'.format(mtc_epo[rank_mtc[i]], rank_mtc[i]))
            os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(
                mtc_epo[rank_mtc[i]], rank_mtc[i], res))
    except:
        pass

    # ------- test ---------------

    test_data = IMG_Folder(opt.excel_path, opt.test_folder)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=opt.batch_size, 
                                                  num_workers=opt.num_workers, 
                                                  pin_memory=True,
                                                  drop_last=False)

    model_test.load_state_dict(torch.load(
        os.path.join(opt.output_dir+opt.model+'_best_model.pth.tar')
        )['state_dict'])
    print('========= best model test result ===========')
    test_MAE,test_CC = test(test_loader,
    model,model_dis,criterion,device,figure=True,figure_name='best_model.png')
    os.system('echo "best valid model TEST MAE mtc:{:.5f}" >> {}'.format(test_MAE.avg, res))
    os.system('echo "best valid model TEST rr mtc:{:.3f}" >> {}'.format(test_CC[0][1], res))


    model_test.load_state_dict(torch.load(
        os.path.join(opt.output_dir+opt.model+'_checkpoint.pth.tar')
        )['state_dict'])
    print('========= last model test result ===========')
    test_MAE,test_CC = test(test_loader,
    model,model_dis,criterion,device,figure=True,figure_name='last_model.png')
    os.system('echo "the last model TEST MAE mtc:{:.5f}" >> {}'.format(test_MAE.avg, res))
    os.system('echo "the last model TEST rr mtc:{:.3f}" >> {}'.format(test_CC[0][1], res))


def train(train_loader, model, dis_model, criterion, optimizer, device,epoch):

    losses = AverageMeter()
    MAE = AverageMeter()

    for i, (img, id,target, male) in enumerate(train_loader):

        # convert male lable to one hot type
        male = torch.unsqueeze(male,1)
        male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
        male = male.to(device).type(torch.FloatTensor)

        input = img.type(torch.FloatTensor).to(device)
        target = torch.from_numpy(np.expand_dims(target,axis=1))
        target = target.type(torch.FloatTensor).to(device)

        predict1 = dis_model(input,male).detach()
        dis_age = discriminate_age(predict1,range=opt.dis_range).to(device)
        # compute residual age label
        residual_age = target - dis_age
        residual_age = residual_age

        # compute prediction residual age
        # loss function is applied to residual age
        model.zero_grad()
        predicted_residual_age = model(input, male, dis_age)
        loss1 = criterion(predicted_residual_age, residual_age)
        
        # compute true age = prediction residual age + dis age
        output_age = predicted_residual_age + dis_age
        a = torch.cat([target,predict1,dis_age,output_age,predicted_residual_age,residual_age],dim=1)
        criterion2 = rank_difference(lbd=10)
        loss2 = criterion2(output_age,target)
        loss = loss2
        mae = metric(output_age.detach(), target.detach().cpu())
        losses.update(loss, img.size(0))
        MAE.update(mae, img.size(0))

        if i % opt.print_freq == 0:
            print(
                    'Epoch: [{0} / {1}]   [step {2}/{3}]   '
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'.format
                    (epoch,opt.epochs, i, len(train_loader), loss=losses,MAE=MAE ))
            print(a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(valid_loader, model, dis_model, criterion, device):
    losses = AverageMeter()
    MAE = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, (input, id,target,male) in enumerate(valid_loader):
            input = input.type(torch.FloatTensor)
            input = input.to(device)

            # one hot encode male lable
            male = torch.unsqueeze(male,1)
            male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
            male = male.type(torch.FloatTensor).to(device)

            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.type(torch.FloatTensor).to(device)

            predict1 = dis_model(input,male).detach()
            dis_age = discriminate_age(predict1,range=opt.dis_range).to(device)
    
            # compute residual age
            residual_age = target - dis_age

            # compute output and loss
            predicted_residual_age = model(input, male, dis_age)
            loss1 = criterion(predicted_residual_age, residual_age)
            output_age = predicted_residual_age + dis_age

            criterion2 = rank_difference(lbd=10)
            loss2 = criterion2(output_age,target)
            loss = loss1+loss2
            mae = metric(output_age.detach(), target.detach().cpu())

            # measure accuracy and record loss
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        print(
            'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
            len(valid_loader), loss=losses, MAE=MAE))

        return losses.avg, MAE.avg

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def save_checkpoint(state, is_best, out_dir, model_name):
    checkpoint_path = out_dir+model_name+'_checkpoint.pth.tar'
    best_model_path = out_dir+model_name+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        # shutil.copyfile(checkpoint_path, model_path)
        torch.save(state, best_model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n\n")

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(w.weight.data)
        w.bias.data.fill_(0.0001)

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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
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
    if os.path.isdir(opt.output_dir): 
        if input("### output_dir exists, rm? ###") == 'y':
            os.system('rm -rf {}'.format(opt.output_dir))
        # set train folder
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        # save train config
    os.system('cp bash_refined.sh {}'.format(opt.output_dir))
    os.system('cp refined_DenseNet.py {}'.format(opt.output_dir))
    os.system('cp refined_CNN.py {}'.format(opt.output_dir))
    os.system('cp refined_train.py {}'.format(opt.output_dir))

    print('=> training from scratch.\n')
    res = os.path.join(opt.output_dir, 'result')
    os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))

    main(res)
