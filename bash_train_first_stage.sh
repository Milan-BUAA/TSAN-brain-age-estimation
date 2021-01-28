#!/bin/bash

paths=/home/liuziyang/workspace/brain_age_prediction

dataset=combine/18
model=$1
loss=$2
batch_size=$3
lbd=$4
beta=$5
num_pair=$6
extra=$7
save_path=./model/combine/${model}_${loss}_lbd_${lbd}_beta_${beta}_np_${num_pair}_${extra}/
label=${paths}/lables/combine.xls

train_data=${paths}/data/NC/${dataset}/train
valid_data=${paths}/data/NC/${dataset}/val
test_data=${paths}/data/NC/${dataset}/test

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0,1 python train.py       \
--batch_size               $batch_size         \
--epochs                   150                 \
--lr                       5e-4                \
--num_workers              15                  \
--print_freq               40                  \
--mix_up                   0.0                 \
--weight_decay             2e-5                \
--loss                     $loss               \
--aux_loss                 both                \
--lbd                      $lbd                \
--beta                     $beta               \
--num_pair                 $num_pair           \
--train_folder             ${train_data}       \
--valid_folder             ${valid_data}       \
--test_folder              ${test_data}        \
--excel_path               ${label}            \
--model                    ${model}            \
--output_dir               ${save_path}        \
