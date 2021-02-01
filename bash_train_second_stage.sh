#! /bin/bash
paths=/home/liuziyang/workspace/brain_age_prediction
dataset=combine/18

dis_range=5
model=ScaleDense
loss=mse
batch_size=4
lbd=10
beta=0.1
first_stage_net=./model/ScaleDense_mse_lbd_10_beta_0.1/ScaleDense_best_model.pth.tar
save_path=./model/second_stage_dis_${dis_range}_${model}/
label=${paths}/lables/combine.xls

train_data=${paths}/data/NC/${dataset}/test
valid_data=${paths}/data/NC/${dataset}/val
test_data=${paths}/data/NC/${dataset}/test

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0     python  train_second_stage.py  \
--batch_size               $batch_size         \
--epochs                   3                   \
--lr                       5e-4                \
--num_workers              15                  \
--print_freq               40                  \
--mix_up                   0.0                 \
--weight_decay             2e-5                \
--loss                     $loss               \
--aux_loss                 ranking             \
--lbd                      $lbd                \
--beta                     $beta               \
--first_stage_net          ${first_stage_net}  \
--train_folder             ${train_data}       \
--valid_folder             ${valid_data}       \
--test_folder              ${test_data}        \
--excel_path               ${label}            \
--model                    ${model}            \
--output_dir               ${save_path}        \
--dis_range                ${dis_range}        \