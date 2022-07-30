"""Entry point for Debugging."""
import argparse
import typing

import torch
from torch import optim

from ConfigReader import create_config
from data.make_dataset import DebugDataset
from Dataset import AminoDS
from distillation.Distillation import Distillation
from Logger import Logger, MLFlowLogger, WandBLogger
from models.Students import Students
from models.Teachers import Teachers


def main(setup: dict = None):
    """Start program.

    :param setup: Optional dictionary with path to data and config,
        as well as wandb boolean. Use if function needs to be run
        in python, leave blank if parameters are passed through
        command line arguments.
    """
    config_path: str = ''
    data_path: str = ''
    logger_bool: bool = False
    # Read Config
    if not setup:
        # TODO typing
        flags = argparse.ArgumentParser(description='knowledge distillation')
        flags.add_argument(  # TODO: Still in use
            '--config_env',
            help='Location of path config file')
        flags.add_argument(
            '--config_exp',
            help='Location of experiments config file',
            required=True)
        flags.add_argument(
            '--path',
            help='Path to data',
            required=True)
        flags.add_argument(
            '--wandb',
            action='store_true',
            default=False)
        args = flags.parse_args()
        config_path = args.config_exp
        data_path = args.path
        logger_bool = args.wandb
    else:
        assert isinstance(setup['config_path'], str)
        assert isinstance(setup['data_path'], str)
        assert isinstance(setup['wandb'], bool)
        config_path = setup['config_path']
        data_path = setup['data_path']
        logger_bool = setup['wandb']

    param: dict = create_config(config_path)

    # Init Logger
    logger: Logger = WandBLogger(param) if logger_bool\
        else MLFlowLogger(param)
    logger.log_params()

    # Init Models
    teacher: torch.nn.Module = Teachers.get_mlp1()
    student: torch.nn.Module = Students.get_debug_student()

    test_data: typing.Any = DebugDataset(40, 10)
    test_data.create_debug_dataset()
    data_train: typing.Any = AminoDS(data_path, dataset_type="train")
    data_test: typing.Any = AminoDS(data_path, dataset_type="test")

    distil = Distillation(
        student=student,
        student_optim=optim.Adam(student.parameters()),
        student_lr=param['student_lr'],
        teacher=teacher,
        teacher_optim=optim.Adam(teacher.parameters()),
        teacher_lr=param['teacher_lr'],
        data_train=data_train,
        data_test=data_test,
        batch_size=param['batch_size'],
        student_epochs=param['student_epochs'],
        teacher_epochs=param['teacher_epochs'],
        meta_epochs=param['meta_epochs'],
        alpha=param['alpha'],
        beta=param['beta'],
        logger=logger,
        t=param['t']
    )

    distil.train_loop(0.5, 0.5)

    test_data.rm_csv()


if __name__ == "__main__":
    main()
