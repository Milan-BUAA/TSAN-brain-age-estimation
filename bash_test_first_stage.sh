#! /bin/bash
model=ScaleDense
test_dirpath=/data/NC/
excel_dirpath=/data/brain_age.xls
model_dirpath=./model/TSNA/
# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0 python prediction_first_stage.py \
--model             ${model}                            \
--batch_size        32                                  \
--num_workers       20                                  \
--output_dir        ${model_dirpath}                    \
--model_name        ${model}_best_model.pth.tar         \
--test_folder       ${test_dirpath}                     \
--excel_path        ${excel_dirpath}                    \
--npz_name          brain_age.npz                       \


