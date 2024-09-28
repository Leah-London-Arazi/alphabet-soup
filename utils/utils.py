import random
import string
from pathlib import Path
import torch
import numpy as np
import logging

from utils.module_logger import ModuleLogger
from utils.defaults import ROOT_LOGGER_NAME


# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
def disable_warnings():
    # tensorflow
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # transformers
    from transformers import logging
    logging.set_verbosity_error()


def random_word(min_len=3, max_len=10):
    length = random.randint(min_len, max_len)  # Random word length
    characters = string.digits + string.ascii_letters + string.punctuation
    word = ''.join(random.choices(characters, k=length))  # Generate a word
    return word


def random_sentence(min_len=3, max_len=10):
    sen_length = random.randint(min_len, max_len)  # Random sentence length
    return " ".join([random_word() for _ in range(sen_length)])


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


def get_logger(name):
    logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return ModuleLogger(logger=logger)


def get_root_logger():
    return logging.getLogger(ROOT_LOGGER_NAME)
