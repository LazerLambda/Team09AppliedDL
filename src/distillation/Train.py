"""Trainer Module."""

from datetime import datetime
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Train:
    """Train Class.

    :method __init__: Initialize Class with all hyperparameters necessary.
    :method train: Train a model in a vanilla pytorch training loop.
    """

    def __init__(
            self,
            module: nn.Module,
            optimizer: optim.Optimizer,
            batch_size: int,
            epochs: int,
            loss: Callable,
            lr: float,
            save_at: int = -1,
            title: str = None,
            cont_train: bool = False,
            device: torch.device = None):
        """Initialize Class.

        Set hyperparameters to class variables, check validity passed arguments
        and set correct device.

        :param module: Model to be trained.
        :param optimizer: Optimizer for training.
        :param batch_size: Batch size used for training.
        :param epochs: Number of epochs to be trained on.
        :param loss: Loss function used in training.
        :param lr: Learning rate parameter.
        :param save_at: integer to determine at which stepsize model is saved
            intermediately. If `save_at` is -1, nothing will be saved during
            training.
        :param title: str: Title for the model's path.
        :param cont_train: Continue training at last checkpoint.
        :param device: Device training should be computed on.
        """
        if title is None:
            title = str(datetime.now())

        assert isinstance(module, nn.Module),\
            "ERROR: `module` must be of class nn.Module."
        # assert isinstance(optimizer, optim.Optimizer),\
        #     "ERROR: `optimizer` must be of class optim.Optimizer."
        assert batch_size > 0, "ERROR: `batch_size` must be larger than 0."
        assert callable(loss), "ERROR: `loss` must be callable."
        assert lr > 0 and lr < 1, "ERROR: `lr` must be in domain (0,1)"

        self.module: nn.Module = module
        self.optimizer: optim.Optimizer = optimizer
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.loss: Callable = loss
        self.lr: float = lr
        self.save_at: int = save_at
        self.title: str = title
        self.cont_train: bool = cont_train

        if device is None:
            self.device: torch.Device =\
                torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.module.to(self.device)

    def save_model(self, model: nn.Module, title: str) -> None:
        """Save Model during Training.
        
        Save intermediate model weights.

        :param title: Title where model will be stored.
        """
        torch.save(model.state_dict(), title)

    def train_teacher(self, data: Dataset, path: str = None):
        """Train Module.

        Training loop to train the defined model.

        :param data: Data the model will be trained on.
        :param path: Path where intermediate results are saved.
        """
        train_data = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            print(f'  Teacher-Epoch: {epoch}')
            for _, [x, y] in enumerate(tqdm(train_data, desc='  Batch')):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                x = x.float()
                prediction = self.module(x)
                loss = self.loss(prediction.squeeze(1), y)
                loss.backward()
                self.optimizer.step()

    def train_student(
            self,
            data: Dataset,
            teacher: nn.Module,
            beta: float,
            path: str = None):
        """Train Module.

        Training loop to train the defined student model.

        :param data: complete training dataset.
        :param teacher: teacher model for creating
            labels on part of the dataset.
        :param beta: determine which proportion the
            dataset should be split in.
        :param path: Path where intermediate results are saved.
        """
        train_data = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            print(f'  Student-Epoch: {epoch}')
            for _, [x, labels] in enumerate(tqdm(train_data, desc='  Batch')):
                x, labels = x.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                proportion_beta: int = int(beta * len(x))
                x_o, x_t = x[0:proportion_beta], x[proportion_beta::]
                labels_o, labels_t = labels[0:proportion_beta],\
                    labels[proportion_beta::]

                x_o, x_t = x[0:proportion_beta].type(torch.FloatTensor),\
                    x[proportion_beta::].type(torch.FloatTensor)
                labels_o, labels_t = labels_o.type(torch.FloatTensor),\
                    labels_t.type(torch.FloatTensor)

                prediction_student_t = self.module(x_t.to(self.device))
                prediction_student_o = self.module(x_o.to(self.device))

                teacher.eval()
                teacher.to(self.device)  # TODO move to teacher file
                prediction_teacher_t = teacher(x_t.to(self.device))
                teacher.train()

                loss = self.loss(
                    prediction_student_t.squeeze(1),
                    prediction_student_o.squeeze(1),
                    prediction_teacher_t.squeeze(1),
                    labels_o)

                loss.backward()
                self.optimizer.step()
