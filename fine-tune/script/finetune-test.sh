#!/bin/bash

# nohup ./finetune-test.sh >> finetune.log 2>&1 &
dataset_name="MulDiGraph"
first_partition=2
end_partition=2
train_first=2
multiple=500
delay=5
strategy="purest"
finetune_dataset_dir="/root/LLaMA-Factory/data/finetune_label_logits_smote"
test_dir="/root/data/clustered_graphs"
pretrain_model_path="/root/autodl-tmp/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B"
adapter_output_dir="/root/autodl-tmp/adapter/"
finetune_output_path=""
result_dir="/root/result"

mkdir -p "$result_dir"


for i in $(seq ${first_partition} ${end_partition}); do
    echo "Processing dataset: ${dataset_name}_delay_${delay}_${i} ..."

    old_model_path=${finetune_output_path}
    if [ "$i" -eq ${first_partition} ]; then
        finetune_model_path=${pretrain_model_path}
        echo "Using pretrained model as initial model: ${finetune_model_path}"
    else
        finetune_model_path=${old_model_path}
        echo "Using previous model as initial model: ${finetune_model_path}"
    fi

    finetune_output_path="/root/autodl-tmp/Llama-3.1-8B_${dataset_name}_delay_${delay}_${i}_SFT"

    
    # finetune_dataset_name="${dataset_name}_delay_5_500_${i},${dataset_name}_delay_5_1000_${i},${dataset_name}_delay_5_2000_${i}"

    # for ((j=${train_first}; j<=${first_partition}; j++)); do
    #     part="${dataset_name}_delay_5_500_${j},${dataset_name}_delay_5_1000_${j},${dataset_name}_delay_5_2000_${j}"
    #     if [ -n "$finetune_dataset_name" ]; then
    #         finetune_dataset_name="${finetune_dataset_name},${part}"
    #     else
    #         finetune_dataset_name="${part}"
    #     fi
    # done

    for ((j=${train_first}; j<=${first_partition}; j++)); do
        part="${dataset_name}_delay_5_${multiple}_${j}"
        if [ -n "$finetune_dataset_name" ]; then
            finetune_dataset_name="${finetune_dataset_name},${part}"
        else
            finetune_dataset_name="${part}"
        fi
    done


    adapter_output_path=${adapter_output_dir}/${i}
    echo "Finetune dataset: ${finetune_dataset_name}"
    echo "Finetune output path: ${finetune_output_path}"
    echo "Adapter output path: ${adapter_output_path}"

    echo "Starting finetune training for ${finetune_dataset_name}..."
    # FORCE_TORCHRUN=1 llamafactory-cli train \
    
    FORCE_TORCHRUN=1 llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path ${finetune_model_path} \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template llama3 \
        --flash_attn fa2 \
        --dataset_dir ${finetune_dataset_dir} \
        --dataset ${finetune_dataset_name} \
        --cutoff_len 2048 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 10000 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 1000 \
        --warmup_steps 100 \
        --optim adamw_torch \
        --packing False \
        --report_to none \
        --output_dir ${adapter_output_path} \
        --fp16 True \
        --plot_loss True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target all \
        --ddp_find_unused_parameters False
        # --deepspeed /root/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json

    if [ $? -ne 0 ]; then
        echo "Error: train for ${finetune_dataset_name} failed."
        exit 1
    fi

    echo "Starting model export for ${finetune_dataset_name}..."
    llamafactory-cli export \
        --model_name_or_path ${fintune_model_path} \
        --adapter_name_or_path ${adapter_output_path} \
        --template llama3 \
        --finetuning_type lora \
        --export_dir ${finetune_output_path} \
        --export_size 2 \
        --export_device cpu \
        --export_legacy_format false

    if [ $? -ne 0 ]; then
        echo "Error: export for ${finetune_dataset_name} failed."
        exit 1
    fi

    python -u /root/prompt/single_expert/llama_finetune_prompt_batch_label_logits.py \
        --model_path ${finetune_output_path} \
        --dataset_name ${dataset_name} \
        --delay ${delay} \
        --strategy ${strategy} \
        --test_dir ${test_dir} \
        --result_dir ${result_dir} \
        --index ${i} \
        --multiple ${multiple}

    # python -u /root/prompt/single_expert/llama_finetune_prompt_batch_label_logits.py \
    #     --model_path "/root/autodl-tmp/Llama-3.1-8B_MulDiGraph_delay_5_12_SFT" \
    #     --dataset_name "MulDiGraph" \
    #     --delay 5 \
    #     --strategy "purest" \
    #     --test_dir "/root/data/clustered_graphs" \
    #     --result_dir "/root/result" \
    #     --index 12 \
    #     --multiple 500

    python -u /root/run_ensemble_llama_finetune_gnn_3.py

    # rm -rf ${adapter_output_path}
    # if [ -n "${finetune_output_path}" ]; then
    #     echo "Deleting old model path: ${finetune_output_path}, adapter output path: ${adapter_output_path}"
    #     rm -rf ${finetune_output_path}
    # fi
    
    if [ $? -ne 0 ]; then
        echo "Error: llama_finetune_prompt_batch for ${finetune_dataset_name} failed."
        exit 1
    fi
    
    echo "Evaluation for ${finetune_dataset_name} completed successfully."
done

echo "All directories processed successfully!"