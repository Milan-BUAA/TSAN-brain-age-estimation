import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description='Train Test split for TSAN brain age estimation')
parser.add_argument('--sdir',default='/data/brain_age_estimation_prep-org/nonlin_brain', type=str, help="source dictionary of preprocessed images")
parser.add_argument('--tdir', default='/data/brain_age_estimation_transfer_learning', type=str, help="target dictionary for train test split")
parser.add_argument('--train', default=0.75,type=float, help="Train set data percentage")
parser.add_argument('--val', default=0.15, type=float, help="Validation set data percentage")

args = parser.parse_args()

def Train_Val_Test_Split(sdir, tdir, train_per=0.75, val_per=0.15):
    files = os.listdir(sdir)
    random.shuffle(files)
    Whole_data_length = len(files)
    
    train_cut_point = int(Whole_data_length * train_per)
    val_cut_point = int(Whole_data_length * (val_per + train_per)) 
    
    train_set = files[:train_cut_point]
    validation_set = files[train_cut_point : val_cut_point ]
    test_set = files[val_cut_point:]
    
    train_tdir = os.path.join(tdir, 'train')
    if not os.path.exists(train_tdir):
        os.makedirs(train_tdir)
    for file in train_set:
        shutil.copyfile(os.path.join(sdir, file),os.path.join(train_tdir, file))
        
    val_tdir = os.path.join(tdir, 'val')
    if not os.path.exists(val_tdir):
        os.makedirs(val_tdir)
    for file in validation_set:
        shutil.copyfile(os.path.join(sdir, file),os.path.join(val_tdir, file))
        
    test_tdir = os.path.join(tdir, 'test')
    if not os.path.exists(test_tdir):
        os.makedirs(test_tdir)
    for file in test_set:
        shutil.copyfile(os.path.join(sdir, file),os.path.join(test_tdir, file))
    
    print(len(train_set), len(validation_set), len(test_set), Whole_data_length)

    
if __name__ == '__main__':
    Train_Val_Test_Split(sdir=args.sdir, tdir=args.tdir, train_per=args.train, val_per=args.val)
    
    
    