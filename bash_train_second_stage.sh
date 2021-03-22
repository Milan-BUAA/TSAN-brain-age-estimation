#! /bin/bash
paths=/home/workspace/brain_age_prediction
dataset=combine/

dis_range=5
model=ScaleDense
loss=mse
batch_size=8
lbd=10
beta=0.1
first_stage_net=./pretrained_model/best_model.pth.tar
save_path=./model/
label=${paths}/lables/combine.xls

train_data=${paths}/data/NC/${dataset}/train
valid_data=${paths}/data/NC/${dataset}/val
test_data=${paths}/data/NC/${dataset}/test
sorter_path=./Sodeep_pretrain_weight/best_lstmla_slen_8.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0     python  train_second_stage.py  \
--batch_size               $batch_size         \
--epochs                   150                 \
--lr                       5e-5                \
--num_workers              15                  \
--print_freq               40                  \
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
--sorter                   ${sorter_path}      \
