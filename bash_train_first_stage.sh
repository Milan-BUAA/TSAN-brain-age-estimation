#!/bin/bash

paths=/home/TSAN-brain-age-estimation/

model=ScaleDense
loss=mse
batch_size=8
lbd=10
beta=0.1
save_path=./pretrained_model/ScaleDense/
label=${paths}/lables/brain_age.xls

train_data=${paths}/data/train
valid_data=${paths}/data/val
test_data=${paths}/data/test

sorter_path=./Sodeep_pretrain_weight/best_lstmla_slen_8.pth.tar
# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0     python train_first_stage.py       \
--batch_size               $batch_size         \
--epochs                   150                 \
--lr                       5e-4                \
--num_workers              15                  \
--print_freq               40                  \
--weight_decay             2e-5                \
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
