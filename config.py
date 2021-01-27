import argparse

parser = argparse.ArgumentParser(description='TSAN-Brain age estimation')
# =========== train setting ================ #
parser.add_argument('--model_name',  default='best_model.pth.tar'       ,type=str)
parser.add_argument('--resume',      default=False, action='store_true' )
parser.add_argument('--resume_step', default='last', help='best ckpt'   ,type=str)
parser.add_argument('--gpus', '-g',  default='0', help='gpu for use.'   ,type=str)

# =========== save path ================ #
parser.add_argument('--output_dir',     default='./model/'              ,type=str)
parser.add_argument('--train_folder',   default='../data/train'         ,type=str)
parser.add_argument('--valid_folder',   default='../data/valid'         ,type=str)
parser.add_argument('--test_folder',    default='../data/test'          ,type=str)
parser.add_argument('--excel_path',     default='../lables/Training.xls',type=str)
parser.add_argument('--first_stage_net',default='./model/best.pth.tar'  ,type=str)
parser.add_argument('--npz_name',       default='test.npz'              ,type=str)
parser.add_argument('--plot_name',      default='test.png'              ,type=str)

#=========== hyperparameter ================ #
parser.add_argument('--model'       ,default='ScaleDense')
parser.add_argument('--num_workers' ,default=10     ,type=int)
parser.add_argument('--batch_size'  ,default=8      ,type=int)
parser.add_argument('--epochs'      ,default=100    ,type=int)
parser.add_argument('--lr'          ,default=1e-3   ,type=float)
parser.add_argument('--print_freq'  ,default=5      ,type=int)
parser.add_argument('--mix_up'      ,default=0.0    ,type=float)
parser.add_argument('--weight_decay',default=5e-4   ,type=float)
parser.add_argument('--use_gender'  ,default=True   ,type=bool)

# =========== loss function ================ #
parser.add_argument('--loss',       default='mse'   ,type=str)
parser.add_argument('--lbd',        default=1       ,type=float)
parser.add_argument('--beta',       default=0.5     ,type=float)
parser.add_argument('--aux_loss',   default='ranking'  ,type=str)
parser.add_argument('--num_pair',   default=40      ,type=int)

# =========== ELSE ================ #
parser.add_argument('--dis_range',        default=5 ,type=int)
args = parser.parse_args()
opt = args