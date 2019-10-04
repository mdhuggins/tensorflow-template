"""Pre-processing for converting raw data into TFRecords"""

# TODO Update parse_raw_data(), write_records() for new project

import os
import json
import random

import numpy as np
import tensorflow as tf

from train.params import params
from components.embeddings import embed, _feature_dim


# ---------- Prepare Data ----------

def parse_raw_data():  # TODO Update for new project
    # Train
    with open(params.raw_data_path + "train.txt", "r") as f:
        lines = f.readlines()
        lines = map(lambda x: x.strip(), lines)

    train_data = []
    for line in lines:
        text, label = line.split(",")

        # Remove non-alpha characters
        text = ''.join(filter(str.isalpha, text))

        # Lowercase and split
        tokens = text.lower().split()

        # Embed with Glove
        tokens = [embed(token) for token in tokens]

        label = np.array([int(label)])
        train_data.append({'tokens': tokens, 'label': label})

    # Test
    with open(params.raw_data_path + "test.txt", "r") as f:
        lines = f.readlines()
        lines = map(lambda x: x.strip(), lines)

    test_data = []
    for line in lines:
        text, label = line.split(",")

        # Remove non-alpha characters
        text = ''.join(filter(lambda c: str.isalpha(c) or str.isspace(c), text))

        # Lowercase and split
        tokens = text.lower().split()

        # Embed with Glove
        tokens = [embed(token) for token in tokens]

        label = np.array([int(label)])
        test_data.append({'tokens': tokens, 'label': label})

    # Automatically determine padding length
    pad_len = max(map(lambda x: len(x['tokens']), train_data+test_data))

    # Pad sequences to length
    pad = [0 for _ in range(_feature_dim)]
    for i in range(len(train_data)):
        sequence = train_data[i]['tokens']

        padded_sequence = sequence + [pad] * pad_len
        truncated_sequence = padded_sequence[:pad_len]

        train_data[i]['tokens'] = truncated_sequence

    for i in range(len(test_data)):
        sequence = test_data[i]['tokens']

        padded_sequence = sequence + [pad] * pad_len
        truncated_sequence = padded_sequence[:pad_len]

        test_data[i]['tokens'] = truncated_sequence

    meta = {'pad_len': pad_len, 'embed_dim': _feature_dim}
    return train_data, test_data, meta


# ---------- Helpers ----------

def _bytes_feature(bytes_list):
    """Supports lists with either bytes or str elements"""
    # Str?
    if len(bytes_list) > 0 and type(bytes_list[0]) is str:
        decoded_value = [str.encode(v) for v in bytes_list]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=decoded_value))

    # Bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))


def _int64_feature(ints_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=ints_list))


def _float_feature(floats_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats_list))


def split_eval(data, shuffle=True):
    # Shuffle data
    random.seed(params.random_seed)
    shuffled = data[:]
    if shuffle:
        random.shuffle(shuffled)

    # Split into train/eval
    cutoff = int(len(shuffled) * params.train_eval_split)

    return shuffled[cutoff:], shuffled[:cutoff]


# ---------- IO ----------

def write_records(data, filename):
    writer = tf.io.TFRecordWriter(params.preprocess_out_dir + filename)

    for d in data:
        example = tf.train.Example(features=tf.train.Features(feature={  # TODO Update Example template for new project
            'tokens': _float_feature(np.array(d['tokens']).flatten()),
            'label': _int64_feature(d['label'])
        }))

        writer.write(example.SerializeToString())

    writer.close()


def write_metadata(data):
    with open(params.preprocess_out_dir + "metadata.json", "w") as f:
        json.dump(data, f)


# ---------- Go! ----------

if __name__ == "__main__":
    if not os.path.exists(params.preprocess_out_dir):
        print("Creating output directory {}...".format(params.preprocess_out_dir))
        os.makedirs(params.preprocess_out_dir)

    print("Parsing raw data...")
    train, test, metadata = parse_raw_data()

    print("Splitting train/eval...")
    train, val = split_eval(train)

    print("Writing TF Records to file...")
    write_records(train, "train.tfrecords")
    write_records(val, "eval.tfrecords")
    write_records(test, "test.tfrecords")

    print("Saving metadata...")
    write_metadata(metadata)

    print("Done!")
