#! /bin/bash
model=ScaleDense
batch_size=32
test_dirpath=./data/test
excel_dirpath=./data/dataset.xls
sorter_path=./TASN/Sodeep_pretrain_weight/best_lstmla_slen_${batch_size}.pth.tar
model_dirpath=./pretrained_model/ScaleDense/

# ------ train and set the parameter
CUDA_VISIBLE_DEVICES=0 python ./TSAN/prediction_first_stage.py \
--model             ${model}                            \
--batch_size        $batch_size                         \
--output_dir        ${model_dirpath}                    \
--model_name        ${model}_best_model.pth.tar         \
--test_folder       ${test_dirpath}                     \
--excel_path        ${excel_dirpath}                    \
--npz_name          brain_age.npz                       \
--sorter            ${sorter_path}                      \

# ============= Hyperparameter Description ============== #
# --model             Deep learning model to do brain age estimation
# --batch_size        Batch size during training process
# --output_dir        Output dictionary, whici will contains training log and model checkpoint
# --model_name        Checkpoint file name
# --test_folder       Test set data path
# --excel_path        Excel file path
# --npz_name          npz file name to store predited brain age
# --sorter            When use ranking loss, the pretrained SoDeep sorter network weight need to be appointed
