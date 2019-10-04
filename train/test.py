""" Evaluate models on test"""

# TODO Update metrics for new project

import logging

import tensorflow as tf

from train.input import test_input_fn, get_config
from train.params import params
from train.models.lstm import model_factory
from train.components.monitoring import MonitorHook
from train.components.logging import setup_logging



def main():
    # Prepare model
    config = tf.estimator.RunConfig(
        model_dir=params.model_dir,
        log_step_count_steps=100,
    )

    classifier = tf.keras.estimator.model_to_estimator(
        keras_model=model_factory(get_config()),
        model_dir=params.model_dir,
        config=config
    )

    print("\n---------- Evaluating on Test ----------")
    # Make eval monitor
    eval_monitor = MonitorHook(len(list(test_input_fn(params.batch_size))), label="Evaluating on Test")  # Hacky way to get number of batches

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda: test_input_fn(params.batch_size),
        hooks=[eval_monitor]
    )

    eval_monitor.cleanup()

    print('\nTest Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    setup_logging()

    main()
