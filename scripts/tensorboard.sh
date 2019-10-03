#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Usage: tensorboard.sh model_name"
else
    MODELDIR=./trained-models/$1
    if [ -d $MODELDIR ]   # for file "if [-f /home/rama/file]"
    then
        tensorboard --logdir $MODELDIR
    else
        echo "Can't find "$MODELDIR
    fi
fi
