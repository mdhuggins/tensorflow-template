"""Hyper-Parameters and Configuration"""

# TODO Update parameters for new project (it's possible that only the model parameters will need to change)

import sys

from absl import flags

# ---------- Pre-Processing ----------

# Paths
flags.DEFINE_string("raw_data_path", "./data/dataset/", "Path to the raw data.")
flags.DEFINE_string("preprocess_out_dir", "./data/records/test/", "Pre-processing output directory.")
flags.DEFINE_string("glove_path", "./glove.pkl", "Pickled Glove embeddings path.")

# Train/Eval
flags.DEFINE_float("train_eval_split", "0.2", "Amount of training data to use for eval.")

# Misc.
flags.DEFINE_integer("random_seed", 1, "Seed to use for random number generation and shuffling.")
flags.DEFINE_string("pad_token", "<PAD>", "Token to pad with.")


# ---------- Training ----------

# Paths
flags.DEFINE_string("model_dir", "./trained-models/default/", "Where to save trained models")


flags.DEFINE_integer("batch_size", 32, "The batch size for training.")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate for training.")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs to train for.")


# ---------- Model ----------
flags.DEFINE_integer("lstm_units", 300, "Number of LSTM units to use.")
flags.DEFINE_float("lstm_dropout", 0.1, "Dropout rate for LSTM")
flags.DEFINE_float("lstm_rec_dropout", 0.1, "Recurrent dropout rate for LSTM")


params = flags.FLAGS
params(sys.argv)
