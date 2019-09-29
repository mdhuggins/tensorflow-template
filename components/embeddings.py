"""Wrapper for Glove Embeddings"""

import pickle

from train.params import params

with open(params.glove_path, 'rb') as f:
    embeddings = pickle.load(f)

_feature_dim = len(list(embeddings.values())[0])


def embed(token):
    oov = [0 for _ in range(_feature_dim)]
    return embeddings.get(token, oov)
