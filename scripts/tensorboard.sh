#!/usr/bin/env bash
# TODO Implement

if [ -z "$1" ]
  then
    echo "Usage: scripts/tensorboard model_name"
else
    tensorboard --logdir ./trained_models/test/$1
fi
