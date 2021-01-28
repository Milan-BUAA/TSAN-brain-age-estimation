#! /bin/bash
paths=/home/liuziyang/workspace/brain_age_prediction
dataset=combine/18
dis_range=5
model=$1
loss=$2
batch_size=$3
lbd=$4
beta=$5
num_pair=$6
extra=$7
save_path=./model/combine/refined_dis_${dis_range}_test_check_${model}_${age_range}/
label=${paths}/lables/combine.xls

train_data=${paths}/data/NC/${dataset}/train
valid_data=${paths}/data/NC/${dataset}/val
test_data=${paths}/data/NC/${dataset}/test

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0 python  second_stage_train.py  \
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
--dis_range ${dis_range}                        \
