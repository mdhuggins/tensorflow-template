# tensorflow-template
A template for Tensorflow 2.0 + Keras projects.

The template has a few key features:

- Training and evaluation data is stored in TFRecords, which are great for large datasets, distributed training, etc.
- Models are implemented in Keras, which greatly reduces the complexity and time required to implement new model architectures.
- Training and evaluation is done using Tensorflow's Estimator framework, which makes things very simple, and supports custom training hooks/monitoring.

## Setup

Python 3 is required. Since this template uses Tensorflow 2.0, which at this time (at the time of writing) is not the default version, it is hightly recommended that you use a virtual environment.

`pip install -r requirements.txt`

## Orientation
- **components/**
    - **embeddings.py** Wrapper for GloVe embeddings use in example
- **data/**
    - **dataset/** Example dataset
    - **records/** Place to save TFRecords
    - **preprocess.py** Pre-processing dataset and writing TFRecords 
- **scripts/**
    - **preprocess.sh** Run pre-processing
    - **tensorboard.sh** Run Tensorboard server
    - **test.sh** Run evaluation on test
    - **train.sh** Run training
- **train/**
    - **components/**
        - **logging.py** Custom logging
        - **monitoring.py** Custom training output
    - **models/**
        - **lstm.py** An example model
    - **input.py**  Interface for reading TFRecords
    - **params.py** Hyperparameters/config definitions and default values
    - **test.py** Top-level testing module
    - **train.py** Top-level training module
- **trained-models/** Where trained models are saved
    - **default/** An example folder model folder
- **glove.pkl**  GloVe embeddings for the example project
- **README.md**  You're here!
- **requirements.txt** Python dependencies

## Updating for Your Project

The majority of code in the template will not need to be changed significantly for most projects. Throughout the template, there are TODOs wherever significant changes are likely needed.

Here's a summary of changes that will probably have to happen:

- In data/preprocess.py, update parse_raw_data() to parse new dataset
- In data/preprocess.py, write_records() with new Example format
- Update training/model parameters (train/params.py, scripts/preprocess.sh, scripts/train.sh)
- In train/models, update or add new models
- In train/input.py, update parse_fn to use new Example format
- In train/train.py and train/test.py, update metrics for new models