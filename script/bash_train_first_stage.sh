#!/bin/bash

model=ScaleDense
loss=mse
batch_size=32
lbd=10
beta=1
save_path=./pretrained_model/ScaleDense/
label=./data/dataset.xls

train_data=./data/train
valid_data=./data/val
test_data=./data/test

sorter_path=./TSAN/Sodeep_pretrain_weight/Tied_rank_best_lstmla_slen_${batch_size}.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0     python ./TSAN/train_first_stage.py       \
--batch_size               $batch_size         \
--epochs                   150                 \
--lr                       1e-3                \
--weight_decay             5e-4                \
--loss                     $loss               \
--aux_loss                 ranking             \
--lbd                      $lbd                \
--beta                     $beta               \
--train_folder             ${train_data}       \
--valid_folder             ${valid_data}       \
--test_folder              ${test_data}        \
--excel_path               ${label}            \
--model                    ${model}            \
--output_dir               ${save_path}        \
--sorter                   ${sorter_path}      \
# ============= Hyperparameter Description ============== #
# --batch_size        Batch size during training process
# --epochs            Total training epochs
# --lr                Initial learning rate
# --weight_decay      L2 weight decay
# --loss              Main loss fuction for training network
# --aux_loss          Auxiliary loss function for training network
# --lbd               The weight between main loss function and auxiliary loss function
# --beta              The weight between ranking loss function and age difference loss function
# --first_stage_net   When training the second stage network, appoint the trained first stage network checkpoint file path is needed
# --train_folder      Train set data path
# --valid_folder      Validation set data path
# --test_folder       Test set data path
# --excel_path        Excel file path
# --model             Deep learning model to do brain age estimation
# --output_dir        Output dictionary, whici will contains training log and model checkpoint
# --dis_range         Discritize step when training the second stage network
# --sorter            When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed