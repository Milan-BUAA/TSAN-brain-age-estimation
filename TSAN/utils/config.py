import argparse

parser = argparse.ArgumentParser(description='TSAN-Brain age estimation')
# =========== save path ================ #
parser.add_argument('--model_name'     ,default='best_model.pth.tar'    ,type=str, help="Checkpoint file name")
parser.add_argument('--output_dir'     ,default='./model/'              ,type=str, help="Output dictionary, whici will contains training log and model checkpoint")
parser.add_argument('--train_folder'   ,default='../data/train'         ,type=str, help="Train set data path ")
parser.add_argument('--valid_folder'   ,default='../data/valid'         ,type=str, help="Validation set data path ")
parser.add_argument('--test_folder'    ,default='../data/test'          ,type=str, help="Test set data path ")
parser.add_argument('--excel_path'     ,default='../lables/Training.xls',type=str, help="Excel file path ")
parser.add_argument('--first_stage_net',default='./model/best.pth.tar'  ,type=str, help="When training the second stage network, appoint the trained first stage network checkpoint file path is needed ")
parser.add_argument('--npz_name'       ,default='test.npz'              ,type=str, help="After inference the trained model in test set, a npz file will be saved in assigned path. So the npz name need to be appointed. ")
parser.add_argument('--plot_name'      ,default='test.png'              ,type=str, help="After inference the trained model in test set, a scatter plot will be saved in assigned path. So the plot name need to be appointed. ")

#=========== hyperparameter ================ #
parser.add_argument('--model'       ,default='ScaleDense',type=str,   help="Deep learning model to do brain age estimation")
parser.add_argument('--num_workers' ,default=8           ,type=int,   help="The number of worker for dataloader")
parser.add_argument('--batch_size'  ,default=8           ,type=int,   help="Batch size during training process")
parser.add_argument('--epochs'      ,default=100         ,type=int,   help="Total training epochs")
parser.add_argument('--lr'          ,default=1e-3        ,type=float, help="Initial learning rate")
parser.add_argument('--schedular', type=str, default='cosine',help='choose the scheduler')
parser.add_argument('--print_freq'  ,default=40           ,type=int,   help="Training log print interval")
parser.add_argument('--weight_decay',default=5e-4        ,type=float, help="L2 weight decay ")
parser.add_argument('--use_gender'  ,default=True        ,type=bool,  help="If use sex label during training")
parser.add_argument('--dis_range'   ,default=5           ,type=int,   help="Discritize step when training the second stage network")

# warmup and cosine_lr_scheduler
parser.add_argument('--warmup_lr_init', type=float, default=5e-7 )
parser.add_argument('--warmup_epoch', type=int, default=0)
parser.add_argument('--min_lr', type=float, default=5e-6 )

# =========== loss function ================ #
parser.add_argument('--loss',       default='mse'       ,type=str,     help="Main loss fuction for training network")
parser.add_argument('--aux_loss',   default='ranking'   ,type=str,     help="Auxiliary loss function for training network")
parser.add_argument('--lbd',        default=10          ,type=float,   help="The weight between main loss function and auxiliary loss function")
parser.add_argument('--beta',       default=1           ,type=float,   help="The weight between ranking loss function and age difference loss function")
parser.add_argument('--sorter',     default='./Sodeep_pretrain_weight/best_lstmla_slen_32.pth.tar', type=str,   help="When use ranking, the pretrained SoDeep sorter network weight need to be appointed")
args = parser.parse_args()
opt = args 