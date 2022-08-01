"""Module to handle configuration .yaml files."""

import errno
import os

import yaml
from easydict import EasyDict


def mkdir_if_missing(directory: str) -> None:
    """Create Diretory if not existing.

    :param directory: Path to directory.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_config(config_file_exp: str) -> dict:
    """Create Configuration from File.

    :param config_file_exp: Path to config file.
    :return: Dictionary with parameter-config.
    """
    # Config for environment path
    root_dir: str = "./models"

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg: EasyDict = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    # create directory for results
    #base_dir: str = os.path.join(root_dir, cfg['teacher'])
    #mkdir_if_missing(base_dir)

    # TODO: saving for each model
    
    
    return cfg
