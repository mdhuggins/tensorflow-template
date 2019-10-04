"""Custom Logging Configuration"""

import logging
from colorama import Fore, Style


# Whitelists applied before blacklists
log_msg_whitelist = ["saving", "loss"]
log_path_whitelist = []
log_path_blacklist = ["session_manager.py", "estimator.py"]
log_msg_blacklist = ["Graph was finalized", "Create CheckpointSaverHook", "Finished evaluation", "Starting evaluation"]


def setup_logging():
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


def log_filter(record):

    for msg in log_msg_whitelist:
        if msg in record.msg:
            return True

    for path in log_path_whitelist:
        if path in record.pathname:
            return True

    for msg in log_msg_blacklist:
        if msg in record.msg:
            return False

    for path in log_path_blacklist:
        if path in record.pathname:
            return False

    return True


class LogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt='%(levelname)s: %(message)s')

    def format(self, record):
        if record.levelno == logging.INFO:
            return record.msg % record.args
        elif record.levelno == logging.WARN:
            msg = record.msg % record.args
            return Fore.YELLOW + "WARNING: " + msg + Style.RESET_ALL
        elif record.levelno == logging.ERROR:
            msg = record.msg % record.args
            return Fore.RED + "ERROR: " + msg + Style.RESET_ALL
        else:
            return super().format(record)
