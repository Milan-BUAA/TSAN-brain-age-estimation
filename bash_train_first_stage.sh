#!/bin/bash

paths=./TSAN-brain-age-estimation/

model=ScaleDense
loss=mse
batch_size=32
lbd=10
beta=1
save_path=./pretrained_model/ScaleDense/
label=${paths}/lables/brain_age.xls

train_data=${paths}/data/train
valid_data=${paths}/data/val
test_data=${paths}/data/test

sorter_path=./Sodeep_pretrain_weight/best_lstmla_slen_${batch_size}.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0     python train_first_stage.py       \
--batch_size               $batch_size         \    # Batch size during training process
--epochs                   150                 \    # Total training epochs
--lr                       1e-3                \    # Initial learning rate
--weight_decay             5e-4                \    # L2 weight decay
--loss                     $loss               \    # Main loss fuction for training network
--aux_loss                 ranking             \    # Auxiliary loss function for training network
--lbd                      $lbd                \    # The weight between main loss function and auxiliary loss function
--beta                     $beta               \    # The weight between ranking loss function and age difference loss function
--train_folder             ${train_data}       \    # Train set data path
--valid_folder             ${valid_data}       \    # Validation set data path
--test_folder              ${test_data}        \    # Test set data path
--excel_path               ${label}            \    # Excel file path
--model                    ${model}            \    # Deep learning model to do brain age estimation
--output_dir               ${save_path}        \    # Output dictionary, whici will contains training log and model checkpoint
--sorter                   ${sorter_path}      \    # When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed
