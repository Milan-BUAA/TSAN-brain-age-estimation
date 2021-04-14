#! /bin/bash
model=ScaleDense
batch_size=32
test_dirpath=/home/TSAN-brain-age-estimation/data/test/
excel_dirpath=/home/TSAN-brain-age-estimation/lables/brain_age.xls
sorter_path=./Sodeep_pretrain_weight/best_lstmla_slen_${batch_size}.pth.tar
model_dirpath=./pretrained_model/second_stage/
first_stage_net=./pretrained_model/ScaleDense/ScaleDense_best_model.pth.tar

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0 python prediction_second_stage.py \
--model             ${model}                             \
--batch_size        $batch_size                          \
--num_workers       20                                   \
--output_dir        ${model_dirpath}                     \
--model_name        ${model}_best_model.pth.tar          \
--test_folder       ${test_dirpath}                      \
--excel_path        ${excel_dirpath}                     \
--npz_name          brain_age.npz                        \
--dis_range         5                                    \
--first_stage_net   ${first_stage_net}                   \
--sorter            ${sorter_path}                       \





