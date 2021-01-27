import os
import torch
import math
import ResNet
import shutil
import denseNet
import numpy as np
import tensorboardX
import torch.nn as nn
from loss import focal
from config import opt
from mix_up import mixup
from load_data import my_DataSet
from torchsummary import summary
from sklearn.metrics import roc_auc_score,mean_absolute_error

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def main():
    best_metric = 100
    best_epoch = None
    print(opt)
    print("----- start train the age prediction model ----------")
    print(" ==========> Using {} processes for data loader.".format(opt.num_workers))

    train_data = my_DataSet(opt.excel_path, opt.train_folder)
    valid_data = my_DataSet(opt.excel_path, opt.valid_folder)
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')

    # balance sample
    if opt.eval_only:
        valid_data = my_DataSet(opt.excel_path, opt.valid_folder)

        valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=opt.batch_size, 
                                               num_workers=opt.num_workers, pin_memory=True,
                                               drop_last=False)

    else:
        train_data = my_DataSet(opt.excel_path, opt.train_folder)
        valid_data = my_DataSet(opt.excel_path, opt.valid_folder)
        # train_targets = torch.tensor([sample[1] for sample in train_data])
        # print(" ==========> get targets of %d data samples" % len(train_targets))
    
        # define data loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                num_workers=opt.num_workers,pin_memory=True,
                                                drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=opt.batch_size, 
                                                num_workers=opt.num_workers, pin_memory=True,
                                                drop_last=False)

    # build and set model 
    model = ResNet.resnet7()
    # model = denseNet.dense_net(4,8,4)
    model = nn.DataParallel(model)
    model.to(device)
    # summary(model,(1,121,145,121))
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MultiLabelSoftMarginLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0005, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 24], gamma=0.1)


    # ------- define tensorboardX --------
    saved_metrics, saved_epos = [], []
    num_epochs = int(opt.epochs)
    sum_writer = tensorboardX.SummaryWriter(opt.outlog_dir)
    print(" > Training is getting started...")
    print(" > Training will take {} epochs...".format(num_epochs))

    if opt.eval_only:
        model =  torch.load(os.path.join(opt.out_dir+opt.model_name))
        validate(valid_loader=valid_loader, model=model, criterion=criterion, device=device)
    else:
        for epoch in range(opt.epochs):
            
            # train for one epoch
            train(train_loader=train_loader, model=model, criterion=criterion,
                  optimizer=optimizer, device=device,epoch=epoch)
            # evaluate on validation set
            valid_loss, valid_mae = validate(valid_loader=valid_loader, model=model, 
                                            criterion=criterion, device=device)
            # learning rate decay
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))

            # write in tensorboard
            sum_writer.add_scalar('valid/1loss', valid_loss, epoch)
            sum_writer.add_scalar('valid/2acc', valid_mae, epoch)
            

            # remember best metric and save checkpoint
            valid_metric = valid_mae
            if valid_metric < best_metric:
                is_best = valid_metric < best_metric
                best_metric = min(valid_metric, best_metric)
                save_checkpoint({
                        'epoch': epoch,
                        'arch': "ResNet9",
                        'state_dict': model.state_dict()},
                        is_best, opt.output_dir,model_name='ResNet9')
                saved_metrics.append(valid_metric)
                saved_epos.append(epoch)
                print('=======>   Best at epoch %d, valid MAE %f\n\n\n' % (epoch, best_metric))
                
def train(train_loader, model, criterion, optimizer, device,epoch):

    losses = AverageMeter()
    ACC = AverageMeter()

    for i, (img, label) in enumerate(train_loader):
        inpt = img.type(torch.FloatTensor)
        
        # label = torch.from_numpy(np.expand_dims(label,axis=1))
        dis_label = torch.from_numpy(label_distribution(label))
        inpt = inpt.to(device)
        dis_label = dis_label.to(device)

        model.zero_grad()
        out = model(inpt)
        dis_label = dis_label.double()
        out = out.double()
        loss = criterion(out, dis_label)
        # acc = metric(
        #         out.detach(), label.detach().cpu())
        losses.update(loss, img.size(0))

        if i % opt.print_freq == 0:
            print(
                    'Epoch: [{0} / {1}]   [step {2}/{3}]   '
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format
                    (epoch,opt.epochs, i, len(train_loader),loss=losses ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(valid_loader, model, criterion, device):
    losses = AverageMeter()
    MAE = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, (input, target) in enumerate(valid_loader):
            input = input.type(torch.FloatTensor)
            input = input.to(device)
            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.to(device)

            # compute output and loss
            output = model(input)

            loss = criterion(output, target)
            mae = metric(output.detach(), target.detach().cpu())

            # measure accuracy and record loss
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        print(
            'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
            len(valid_loader), loss=losses, MAE=MAE))

        return losses.avg, MAE.avg

def metric(output, target):
    batch_size = target.size(0)
    target = target.data.numpy()
    pred = torch.max(output.cpu(), dim=1)[1]  # .max() return (max, max_indices)
    pred = pred.data.numpy()
    acc = float((pred == target).astype(int).sum()) / batch_size

    return acc

def save_checkpoint(state, is_best, out_dir, model_name, filename='_checkpoint.pth.tar'):
    checkpoint_path = out_dir+model_name+'_lr='+str(opt.lr)+filename
    model_path = out_dir+model_name+'_lr='+str(opt.lr)+'_best_model.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, model_path)
        print("=======>   This is the best model !!! It has been saved!!!!!!\n")

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

def label_distribution(label,nb_point=5):
    dis_label = []
    
    for age in label:
        mean = 0
        cov = 0.001
        noise = np.random.normal(mean,cov,nb_point)
        
        u = age  # 均值μ
        sig = math.sqrt(0.2)  # 标准差δ

        x = np.linspace(u - 3 * sig, u + 3 * sig, nb_point)
        a = int(x[0]-0-1)
        b = int(100-x[-1])
        
        ay = np.zeros(a)
        by = np.zeros(b)
        x = torch.from_numpy(x)
        noise = torch.from_numpy(noise)
        y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
        y_sig += noise 
        s = y_sig.sum()
        part = []
        for i in y_sig:
            part.append(i/s)
        part = np.array(part)
        
        part = np.concatenate([ay,part,by],axis=0)
        dis_label.append(part)

    dis_label = np.array(dis_label)
        
    return dis_label

if __name__ == "__main__":
    main()
