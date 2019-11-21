#!/usr/bin/env bash

# TODO Update parameters for new project

# Parameters (defaults are defined in params.py)
NUM_EPOCHS=10

if [ -z "$1" ]
  then
    echo "Usage: train.sh model_name"
else
    MODELDIR=./trained-models/$1
    python -m train.train \
    --model_dir $MODELDIR \
    --num_epochs $NUM_EPOCHS
fi