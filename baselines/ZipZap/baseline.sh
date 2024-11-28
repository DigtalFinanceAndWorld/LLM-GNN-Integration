#!/bin/bash

dataset_dir="../data/MulDiGraph/delay_5"
result_dir="../result/MulDiGraph/delay_5"

mkdir -p "$result_dir"

python gen_pretrain_data.py --data_dir="$dataset_dir/1"
if [ $? -ne 0 ]; then
    echo "Error generating pre-training data for $dataset_dir/1"
    exit 1
fi

python run_pretrain.py --data_dir="$dataset_dir/1" --bert_config_file="./zipzap_config.json"
if [ $? -ne 0 ]; then
    echo "Error during fine-tuning and evaluation for $dataset_dir/1"
    exit 1
fi

checkpoint_file="$dataset_dir/1/ckpt_dir/checkpoint"
model_path=$(grep 'model_checkpoint_path:' "$checkpoint_file" | awk -F'"' '{print $2}')
pretrain_checkpoint="$dataset_dir/1/ckpt_dir/$model_path"
echo "pretrain_checkpoint: pretrain_checkpoint"
if [ -z "$pretrain_checkpoint" ]; then
    echo "Failed to find model from pretrain_checkpoint file."
    exit 1
fi

folder_list=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort -V)

for sub_dir_name in $folder_list; do
    data_dir="$dataset_dir/$sub_dir_name"
    if [ -d "$data_dir" ]; then
        echo "Processing directory: $data_dir"

        python gen_finetune_data.py --data_dir="$data_dir" --vocab_file_name="$dataset_dir/1/vocab"
        if [ $? -ne 0 ]; then
            echo "Error generating fine-tuning data for $data_dir"
            exit 1
        fi

        python run_finetune.py --data_dir="$data_dir" --result_dir="$result_dir" --init_checkpoint="$pretrain_checkpoint" --vocab_file_name="$dataset_dir/1/vocab" --bert_config_file="./zipzap_config.json" --index="$sub_dir_name" --last_index="${folder_list[-1]}"
        if [ $? -ne 0 ]; then
            echo "Error during fine-tuning and evaluation for $data_dir"
            exit 1
        fi

        echo "Successfully processed $data_dir"
        echo "------------------------------"
    fi
done

echo "All directories processed successfully!"
