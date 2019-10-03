#!/usr/bin/env bash

# TODO Update parameters for new project

# Parameters (defaults are defined in params.py)
RAW_DATA_PATH=./data/dataset/
PREPROCESS_OUT_DIR=./data/records/test/
TRAIN_EVAL_SPLIT=0.2

python -m data.preprocess \
    --raw_data_path $RAW_DATA_PATH \
    --preprocess_out_dir $PREPROCESS_OUT_DIR \
    --train_eval_split $TRAIN_EVAL_SPLIT \
