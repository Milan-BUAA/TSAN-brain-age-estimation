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
--model             ${model}                             \ # Deep learning model to do brain age estimation
--batch_size        $batch_size                          \ # Batch size during training process
--output_dir        ${model_dirpath}                     \ # Output dictionary, whici will contains training log and model checkpoint
--model_name        ${model}_best_model.pth.tar          \ # Checkpoint file name
--test_folder       ${test_dirpath}                      \ # Test set data path
--excel_path        ${excel_dirpath}                     \ # Excel file path
--npz_name          brain_age.npz                        \ # npz file name to store predited brain age
--dis_range         5                                    \ # Discritize step when training the second stage network
--first_stage_net   ${first_stage_net}                   \ # trained first stage network checkpoint file path
--sorter            ${sorter_path}                       \ # When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed





