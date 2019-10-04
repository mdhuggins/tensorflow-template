""" Train and evaluate models """

# TODO Update metrics for new project

import os
import json
from pathlib import Path
import logging

import tensorflow as tf

from train.input import train_input_fn, eval_input_fn, get_config
from train.params import params
from train.models.lstm import model_factory
from train.components.monitoring import MonitorHook
from train.components.logging import setup_logging


def check_model_directory():
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


def save_config():
    flags_list = [f.serialize() for f in params._get_flags_defined_by_module("train.params")]
    flags_dict = dict([(f.split("=")[0].replace("--", ""), f.split("=")[1]) for f in flags_list])
    config_dict = get_config()
    all_params_dict = {"params": flags_dict, "pre-processing metadata": config_dict}

    config_out_path = os.path.join(params.model_dir, "config.json")
    print("Saving training config to {}".format(config_out_path))

    # If config already saved, append new config to existing file
    if os.path.exists(config_out_path):
        with open(config_out_path, "r") as f:
            existing_dump = json.load(f)

        with open(config_out_path, "w") as f:
            json.dump(existing_dump + [all_params_dict], f, indent=4, sort_keys=True)
    else:
        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)
        with open(config_out_path, "w") as f:
            json.dump([all_params_dict], f, indent=4, sort_keys=True)


def main():
    # Check if model already exists in model directory
    check_model_directory()

    # Write metadata/params to file
    save_config()

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

    print("\n---------- Training ----------")
    # Make training monitor
    monitor = MonitorHook(len(list(train_input_fn(params.batch_size))), label="Training")  # Hacky way to get number of batches

    # Train the model
    classifier.train(
        input_fn=lambda: train_input_fn(params.batch_size),
        hooks=[monitor]
    )

    monitor.cleanup()

    print("\n---------- Evaluating on Train ----------")
    # Make eval monitor
    eval_monitor = MonitorHook(len(list(train_input_fn(params.batch_size))), label="Evaluating on Train")  # Hacky way to get number of batches

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda: train_input_fn(params.batch_size),
        hooks=[eval_monitor]
    )

    eval_monitor.cleanup()

    print('\nTraining Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    print("\n---------- Evaluating on Eval ----------")
    # Make eval monitor
    eval_monitor = MonitorHook(len(list(eval_input_fn(params.batch_size))), label="Evaluating on Eval")  # Hacky way to get number of batches

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(params.batch_size),
        hooks=[eval_monitor]
    )

    eval_monitor.cleanup()

    print('\nEval Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    setup_logging()
    main()
