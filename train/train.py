""" Train and evaluate models """

import os
from pathlib import Path
import logging

import tensorflow as tf

from train.input import train_input_fn, eval_input_fn, get_config
from train.params import params
from train.models.lstm import model_factory
from train.components.monitoring import MonitorHook
from train.components.logging import LogFormatter, log_filter


def main():
    # Check if model already exists in model directory
    if os.path.exists(params.model_dir):
        print("A saved model already exists in {}, "
              "do you want to continue training the saved model (continue) or choose a new model directory (new)?".format(params.model_dir))

        invalid_response = True
        while invalid_response:
            invalid_response = False

            response = input("continue/new > ")
            if response == "continue":
                print("Continuing to train existing model.")
            elif response == "new":
                invalid_name = True
                while invalid_name:
                    invalid_name = False
                    prefix = str(Path(params.model_dir).parent) + "/"
                    new_name = input("New model directory: {}".format(prefix))
                    if os.path.exists(prefix+new_name):
                        print("{} already exists.".format(prefix+new_name))
                        invalid_name = True
                    else:
                        print("Using {} as new model directory.".format(prefix+new_name))
                        params.model_dir = prefix+new_name
            else:
                print("Invalid response. Please choose from \"continue\" or \"new\".")
                invalid_response = True

    config = tf.estimator.RunConfig(
        model_dir=params.model_dir,
        log_step_count_steps=100,
    )

    classifier = tf.keras.estimator.model_to_estimator(
        keras_model=model_factory(get_config()),
        model_dir=params.model_dir,
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

# TODO save params/config with model
