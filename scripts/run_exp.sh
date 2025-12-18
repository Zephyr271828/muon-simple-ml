#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=logs/run_exp_%j.out
#SBATCH --error=logs/run_exp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --time=4:00:00


optimizers=(
    adamw
    muon 
    sgd
)

lrs=(
    0.0001
    0.0003
    0.001
    0.003
    0.01
)

run_exp() {
    model=$1
    dataset=$2
    limit=${3:-10000}
    hidden_size=${4:-512}
    for optimizer in "${optimizers[@]}"; do
        for lr in "${lrs[@]}"; do
            python experiments/main.py \
                --model $1 \
                --lr $lr \
                --wd 0.01 \
                --hidden_size $hidden_size \
                --dataset $2 \
                --optimizer $optimizer \
                --limit $limit
        done
    done

    python experiments/plot.py \
        --log_pattern "logs/*${1}*${2}*${limit}*.log"  \
        --save_path "plots/${1}_${2}_${limit}.png" 
}

for model in olmo qwen llama; do
    run_exp $model openwebtext-100k 10000
done
# run_exp qwen openwebtext-100k 10000 512 
# run_exp llama openwebtext-100k 10000 2048
# for model in resnet18 mobilenet_v2 efficientnet_b0 convnext_tiny; do
#     run_exp $model imagenet 1000000
# done

# run_exp resnet imagenet 1000000
# run_exp quadratic dummy 1000