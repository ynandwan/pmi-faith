
# Train

#!/bin/sh
model_name_or_path="facebook/bart-large"


######     Train multidoc2dial    ###############
DATA_DIR='data/multidoc2dial'
BASE_DIR='trained-models/multidoc2dial/'
OUT_DIR="${BASE_DIR}/bart-large/lr-5"

mkdir -p ${OUT_DIR}
train_cmd="python models/dialog.py --model_name_or_path ${model_name_or_path} --do_train --output_dir ${OUT_DIR} --warmup_ratio 0.04 --max_seq_length 512 --gradient_accumulation_steps 4 --num_train_epochs 50 --filter 0 --filter_after_truncate --train_batch_size 16 --train_dataset_path ${DATA_DIR}/train.json --eval_dataset_path ${DATA_DIR}/validation.json --learning_rate 6.25e-5 --patience 20"

echo $train_cmd
eval $train_cmd




# Train FD

BASE_DIR="trained-models/fd"
OUT_DIR="${BASE_DIR}/bart-large/lr-5"

export train_cmd
train_cmd="python models/dialog.py --model_name_or_path ${model_name_or_path} --do_train --output_dir ${OUT_DIR}  --warmup_ratio 0.04 --max_seq_length 512 --max_history 2 --gradient_accumulation_steps 4 --num_train_epochs 10 --train_batch_size 32 --loss_truncation 0 --filter 0 --learning_rate 6.25e-5"

echo $train_cmd
eval $train_cmd


# TRAIN Topical Chat

BASE_DIR="trained-models/tc"
OUT_DIR="${BASE_DIR}/bart-large/lr-5"

train_cmd="python models/dialog.py --model_name_or_path ${model_name_or_path} --do_train --output_dir ${OUT_DIR} --warmup_ratio 0.04 --max_seq_length 512 --max_history 2 --gradient_accumulation_steps 4 --num_train_epochs 10 --train_batch_size 16 --train_dataset_path data/tc_nopersonal/tc_train.json --eval_dataset_path data/tc_nopersonal/tc_valid_rare.json --loss_truncation 0 --filter 0 --learning_rate 6.25e-5"
echo $train_cmd
eval $train_cmd
