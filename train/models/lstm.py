""""A simple LSTM"""

# TODO Implement new models for new project

from train.params import params

import tensorflow as tf


def model_factory(config):
    pad_len = config['pad_len']
    embed_dim = config['embed_dim']

    inputs = tf.keras.Input(shape=(pad_len, embed_dim))

    lstm = tf.keras.layers.LSTM(params.lstm_units,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                dropout=params.lstm_dropout,
                                recurrent_dropout=params.lstm_rec_dropout,
                                input_shape=(pad_len, embed_dim)
                                )(inputs)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(lstm)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    loss = tf.keras.losses.hinge
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
