#!/usr/bin/env bash

# Parameters (defaults are defined in params.py)
MODEL_DIR="./trained-models/default/"

if [ -z "$1" ]
  then
    echo "Usage: test.sh model_name"
else
    MODELDIR=./trained-models/$1
    if [ -d $MODELDIR ]   # for file "if [-f /home/rama/file]"
    then
        python -m train.test \
            --model_dir $MODEL_DIR
    else
        echo "Can't find "$MODELDIR
    fi
fi