import random
import string
from pathlib import Path
import torch
import numpy as np
import logging
from logging import _nameToLevel
import sys
from datetime import datetime
from utils.module_logger import ModuleLogger
from utils.defaults import ROOT_LOGGER_NAME, DEFAULT_RANDOM_SENTENCE_LENGTH


def disable_warnings():
    # filter FutureWarning
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # textattack
    class NoWarningsFilter(logging.Filter):
        def filter(self, record):
            # Filter out warnings (i.e., records with WARNING level)
            if record.levelno == logging.WARNING:
                return False
            return True

    textattack_install_logger = logging.getLogger("textattack.shared.utils.install")
    textattack_install_logger.addFilter(NoWarningsFilter())

    # tensorflow
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # transformers
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_warning()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def random_word(min_len=3, max_len=10):
    length = random.randint(min_len, max_len)  # Random word length
    characters = string.digits + string.ascii_letters + string.punctuation
    word = ''.join(random.choices(characters, k=length))  # Generate a word
    return word


def random_sentence(sentence_len=DEFAULT_RANDOM_SENTENCE_LENGTH):
    return " ".join([random_word() for _ in range(sentence_len)])


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def create_dir(dir_name):
    directory = Path(dir_name)
    directory.mkdir(parents=True, exist_ok=True)
    return dir_name


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_logger(name):
    logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return ModuleLogger(logger=logger)


def init_logger(level_name):
    try:
        level = _nameToLevel[level_name]
    except KeyError:
        raise ValueError("Invalid logging level")

    logging.basicConfig(level=level)
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level=level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s\n%(message)s\n',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_root_logger():
    return logging.getLogger(ROOT_LOGGER_NAME)


def get_escaped_model_name(model_name):
    return model_name.replace("/", "_")
