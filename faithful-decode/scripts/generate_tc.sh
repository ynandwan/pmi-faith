#!/bin/bash

dataset="tc"
BASE_DIR="trained-models/${dataset}/bart-large/lr-5"
DATA_DIR="data/tc_nopersonal"

for do_sample in 0 1; do
    for maskp in 0.6 0.75 0.9 1.0; do
      for pmi_weight in 0.0 0.25 0.5 1.0; do
        cmd="python models/pmi_generate.py --model_name_or_path ${BASE_DIR}/best_model --output ${BASE_DIR}/generated/best_model --dataset_path ${DATA_DIR}/tc_test_rare.json --batch_size 32 --top_p ${maskp} --pmi_decode --pmi_weight ${pmi_weight} --do_sample ${do_sample}"
        echo $cmd
        #eval $cmd
      done
    done
done
