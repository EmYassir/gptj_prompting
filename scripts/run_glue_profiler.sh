#!/bin/bash
echo "############## Starting evaluation script... ";

## General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=4

# Distributed env
export RANK=0
export UPPER_BOUND=63000                     #Upper Range
export LOWER_BOUND=20000                     #Lower Range
export DIFF=$((UPPER_BOUND-LOWER_BOUND+1))   #+1 to inlcude upper limit
export MASTER_PORT=$(($(($RANDOM%$DIFF))+LOWER_BOUND))

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


## Distributed training
export CUDA_VISIBLE_DEVICES=1


## Model
export MODEL_TYPE=gpt2
export MODEL_NAME=/media/data/yassir/original_models/gpt2
#export MODEL_TYPE=EleutherAI/gpt-j-6B
#export MODEL_NAME=EleutherAI/gpt-j-6B

"""
## Training hyper-parameters
export EPOCHS=3
export MAX_SEQ_LEN=128
export TRAIN_BATCH_SIZE=4
export GRAD_ACC=1
"""
## Evaluation hyper-parameters
export MAX_SEQ_LEN=1024
export NUM_EXAMPLES=10
export EVAL_BATCH_SIZE=4


## Datasets
export TASK_NAME=MRPC
export DATASET_NAME=glue
export METHOD=few-shot
export DATASET_CONFIG_NAME=glue


## Training settings
export RUNNER=/home/yassir/gptj_prompt/cli.py
export DATA_DIR=/media/data/yassir/datasets/$DATASET_NAME/$TASK_NAME/
export OUTPUT_DIR=/media/data/yassir/output/$MODEL_NAME/$DATASET_NAME/$TASK_NAME/$METHOD


############## prompt and prompt descrirption
# format:
# task_name: pattern_ids numbers
# task_name: priming_description

# rte: 0 1 2 3 4 5 6 7
# rte: "Does sentence A entail sentence B ?"
# CoLA: 0 1 2
# CoLA: "Is the sentence grammatical ?"
# SST-2: 0 1 2 3 4 5 6 7 8
# SST-2: "Is the moview review positive or negative ?"
# mnli: 0 1 2 3 4 5 6 7 8 9
# mnli: "Does sentence A entail sentence B ?"
# qnli: 0 1 2 3 4
# qnli: "Does sentence B contain the answer to the question in sentence A ?"
# mrpc: 0 1 2 3 4 5 6 7
# mrpc: "Is the sentence B a paraphrase of sentence A ?"
# qqp: 0 1 2 3 4 5 6 7
# qqp: "Are the two questions similar ?"
################

python $RUNNER \
--fp16 \
--method pet \
--pattern_ids 1 \
--data_dir $DATA_DIR \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME \
--pet_per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--task_name $TASK_NAME \
--output_dir $OUTPUT_DIR \
--do_eval \
--no_distillation \
--pet_repetitions 1 \
--pet_max_seq_length $MAX_SEQ_LEN \
--priming \
--priming_num $NUM_EXAMPLES 



