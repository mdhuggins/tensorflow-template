"""Input for training and evaluation"""

# TODO Update parse_fn for new project

import json

import tensorflow as tf

from train.params import params


def get_config():
    """Load dataset's config output from pre-processing"""
    with open(params.preprocess_out_dir+"metadata.json", 'r') as f:
        return json.load(f)


config = get_config()


def parse_fn(example):
    pad_len = config['pad_len']
    embed_dim = config['embed_dim']

    features = {
        'label': tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'tokens': tf.io.FixedLenSequenceFeature((), tf.float32, allow_missing=True)
    }

    parsed = tf.io.parse_single_example(serialized=example, features=features)

    parsed['label'] = tf.reshape(parsed['label'], [])
    parsed['tokens'] = tf.reshape(parsed['tokens'], [pad_len, embed_dim])

    # Make sure this matches model
    output_features = {
        'input_1': parsed['tokens']
    }

    return output_features, parsed['label']


def train_input_fn(batch_size):
    """An input function for training"""
    # Load data
    filenames = [params.preprocess_out_dir+"train.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse
    dataset = dataset.map(parse_fn)

    # Shuffle and batch the data
    dataset = dataset.shuffle(params.random_seed).batch(batch_size).repeat(count=params.num_epochs)

    return dataset


def eval_input_fn(batch_size):
    """An input function for evaluation"""
    # Load data
    filenames = [params.preprocess_out_dir+"eval.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse
    dataset = dataset.map(parse_fn)

    # Shuffle and batch the data
    dataset = dataset.shuffle(params.random_seed).batch(batch_size)

    return dataset


def test_input_fn(batch_size):
    """An input function for evaluation on the test set"""
    # Load data
    filenames = [params.preprocess_out_dir+"test.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Parse
    dataset = dataset.map(parse_fn)

    # Shuffle and batch the data
    dataset = dataset.shuffle(params.random_seed).batch(batch_size)

    return dataset
