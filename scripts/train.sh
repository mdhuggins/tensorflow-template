#!/usr/bin/env bash

# TODO Update parameters for new project

# Parameters (defaults are defined in params.py)
MODEL_DIR="./trained-models/default/"
NUM_EPOCHS=10

python -m train.train \
    --model_dir $MODEL_DIR \
    --num_epochs $NUM_EPOCHS
