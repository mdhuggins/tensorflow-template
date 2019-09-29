""" Train and evaluate models """

import logging

import tensorflow as tf

from train.input import train_input_fn, eval_input_fn, get_config
from train.params import params
from train.models.lstm import model_factory
from train.components.monitoring import MonitorHook
from train.components.logging import LogFormatter, log_filter


def main():
    config = tf.estimator.RunConfig(
        model_dir=params.model_dir,
        log_step_count_steps=100,
    )

    classifier = tf.keras.estimator.model_to_estimator(
        keras_model=model_factory(get_config()),
        model_dir=params.model_dir,  # TODO Make sure training from scratch
        config=config
    )

    print("\n---------- Training ----------")
    # Make training monitor
    monitor = MonitorHook(len(list(train_input_fn(params.batch_size))), label="Training")  # Hacky way to get number of batches

    # Train the model
    classifier.train(
        input_fn=lambda: train_input_fn(params.batch_size),
        hooks=[monitor]
    )

    monitor.cleanup()

    print("\n---------- Testing on Train ----------")
    # Make eval monitor
    eval_monitor = MonitorHook(len(list(train_input_fn(params.batch_size))),
                               label="Testing on Train")  # Hacky way to get number of batches

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda: train_input_fn(params.batch_size),
        hooks=[eval_monitor]
    )

    eval_monitor.cleanup()

    print('\nTraining Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    print("\n---------- Testing on Eval ----------")
    # Make eval monitor
    eval_monitor = MonitorHook(len(list(eval_input_fn(params.batch_size))), label="Testing on Eval")  # Hacky way to get number of batches

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(params.batch_size),
        hooks=[eval_monitor]
    )

    eval_monitor.cleanup()

    print('\nEval Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    # Setup logging
    tf_log = logging.getLogger("tensorflow")
    tf_log.setLevel(logging.INFO)
    tf_log.addFilter(log_filter)

    # Add logging formatting
    tf_log.removeHandler(tf_log.handlers[0])
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(LogFormatter())
    tf_log.addHandler(ch)

    main()

# TODO Naming for model folders
# TODO save params/config with model
