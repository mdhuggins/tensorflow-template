"""Monitoring hook for training"""

from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.training import session_run_hook


class MonitorHook(session_run_hook.SessionRunHook):
    def __init__(self, num_train_steps, label=""):
        self.progress_bar = tqdm(total=num_train_steps, unit="Batch", desc=label)

    def after_run(self, run_context, run_values):
        self.progress_bar.update()

    def cleanup(self):
        self.progress_bar.close()
