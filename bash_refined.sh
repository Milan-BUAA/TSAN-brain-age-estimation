#! /bin/bash

paths=/home/lzy/Data/age
age_range=18
dataset=combine
model=DenseNet
dis_range=5
save_path=./model/combine/refined_dis_${dis_range}_test_check_${model}_${age_range}/
label=${paths}/lables/${age_range}_${dataset}.xls

train_data=${paths}/data/NC/${dataset}/${age_range}/train
valid_data=${paths}/data/NC/${dataset}/${age_range}/val
test_data=${paths}/data/NC/${dataset}/${age_range}/test

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0,1,2,3 python  refined_train.py                                \
--batch_size 32                                 \
--epochs 200                                    \
--lr 1e-5                                       \
--num_workers 20                                \
--print_freq 10                                 \
--train_folder ${train_data}                    \
--valid_folder ${valid_data}                    \
--test_folder ${test_data}                      \
--excel_path ${label}                           \
--model ${model}                                \
--output_dir ${save_path}                       \
--loss MAE                                      \
--lbd 10                                        \
