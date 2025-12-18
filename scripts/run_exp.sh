#!/bin/bash


run_exp() {
    model=$1
    dataset=$2
    limit=${3:-10000}
    hidden_size=${4:-512}
    for optimizer in adamw muon sgd; do
        for lr in 0.0001 0.0003 0.001 0.003 0.01; do
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

# run_exp qwen openwebtext-100k 10000 512 
# run_exp llama openwebtext-100k 10000 2048
run_exp lstm ptb 1000000
run_exp resnet cifar100 1000000
# run_exp resnet imagenet 1000000
# run_exp quadratic dummy 1000