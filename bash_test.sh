#! /bin/bash
model=CNN
test_dirpath=/data/ziyang/age/data/stroke/nonlinear_brain/
excel_dirpath=/data/ziyang/age/lables/C3_brain_age.xls
model_dirpath=./model/combine/CNN_mse_lbd_20_beta_0.1_np_60_kfold3_/
# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0,1 python  prediction.py          \
--model             ${model}                            \
--batch_size        32                                  \
--num_workers       20                                  \
--output_dir        ${model_dirpath}                    \
--model_name        ${model}_best_model.pth.tar         \
--test_folder       ${test_dirpath}                     \
--excel_path        ${excel_dirpath}                    \
--npz_name          C3_brain_age_fold3.npz                    \


