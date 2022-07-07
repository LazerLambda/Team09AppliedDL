"""Entry point for Debugging."""
import argparse

import torch
from torch import optim

from ConfigReader import create_config
from data.make_dataset import DebugDataset
from Dataset import AminoDS
from distillation.Distillation import Distillation
from Logger import Logger
from models.Students import Students
from models.Teachers import Teachers


def main():
    """Start program."""
    # Read Hyperparam.yaml
    flags = argparse.ArgumentParser(description='knowledge distillation')
    flags.add_argument(
        '--config_env',
        help='Location of path config file')
    flags.add_argument(
        '--config_exp',
        help='Location of experiments config file')

    args = flags.parse_args()
    param: dict = create_config(args.config_exp)

    # Init Logger
    logger: Logger = Logger()
    logger.log_params(param)

    # Init Models
    teacher: torch.nn.Module = Teachers.get_transformer()
    student: torch.nn.Module = Students.get_debug_student()

    test_data: any = DebugDataset(40, 5)
    test_data.create_debug_dataset()
    data_train: any = AminoDS('TEST_DATA.gzip', dataset_type="train")
    data_test: any = AminoDS('TEST_DATA.gzip', dataset_type="test")

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
