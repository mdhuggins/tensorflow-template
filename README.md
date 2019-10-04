# tensorflow-template
A template for Tensorflow 2.0 + Keras projects

## Updating for Your Project

The majority of code in the template will not need to be changed significantly for most projects. Throughout the template, there are TODOs wherever significant changes are likely needed.

Here's a summary of changes that will probably have to happen:

- In data/preprocess.py, update parse_raw_data() to parse new dataset
- In data/preprocess.py, write_records() with new Example format
- Update training/model parameters (train/params.py, scripts/preprocess.sh, scripts/train.sh)
- In train/models, update or add new models
- In train/input.py, update parse_fn to use new Example format
- In train/train.py and train/test.py, update metrics for new models