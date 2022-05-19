import numpy as np
import random
import torch
from argparse import ArgumentParser
import os


def convert_underscore_to_camelcase(word: str):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def record_args(args: ArgumentParser) -> str:
    """
        Print the arguments and creates a name for the run

        :param args: ArgumentParser
        :return: name_of_run: str
    """
    name_of_run = ''

    for arg in vars(args):
        arg_value = getattr(args, arg)
        print(f"{arg}: {arg_value}", flush=True)

        # ignore optimization args
        if arg in ['lr', 'epochs', 'decay']:
            continue

        # ignore args that are set to None
        if arg_value is None:
            continue

        # dataset str names are already pre_defined
        if arg == 'dataset_name':
            name_of_run += arg_value.string().capitalize()
            continue

        name_of_run += convert_underscore_to_camelcase(word=arg) + str(arg_value) + '_'

    print(flush=True)
    return name_of_run[:-1]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_git_path() -> os.path:
    """
    a get function that returns the path to the git dir

    :return: git_dir: os.path
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    git_dir = os.path.dirname(current_dir)
    return git_dir
