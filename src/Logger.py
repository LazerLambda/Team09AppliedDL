"""Logger Class."""

from abc import ABC, abstractmethod

import mlflow

import wandb


class Logger(ABC):
    """Logger Class.

    :method __init__: Initialize class.
    :method log_params: Log hyperparameters.
    :method log_metrics: Log metrics.
    """

    @abstractmethod
    def __init__(self, params: dict):
        """Initialize Class.

        :param params: Dictionary for hyperparameters to be saved.
        """
        pass

    @abstractmethod
    def log_params(self) -> None:
        """Log Hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict) -> None:
        """Log Metrics.

        :param metrics: Dictionary including description of metrics
            and metric-values.
        """
        pass


class MLFlowLogger(Logger):
    """Logger Class for MLFlow.

    :method __init__: Initialize class.
    :method log_params: Log hyperparameters.
    :method log_metrics: Log metrics.
    """

    def __init__(self, params: dict):
        """Initialize Class.

        :param params: Dictionary for hyperparameters to be saved.
        """
        self.params: dict = params

    def log_params(self) -> None:
        """Log Hyperparameters."""
        for key, value in self.params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict) -> None:
        """Log Metrics.

        :param metrics: Dictionary including description of metrics
            and metric-values.
        """
        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


class WandBLogger(Logger):
    """Logger Class for Weights and Biases.

    :method __init__: Initialize class.
    :method log_params: Log hyperparameters.
    :method log_metrics: Log metrics.
    """

    def __init__(self, params: dict):
        """Initialize Class.

        :param params: Dictionary for hyperparameters to be saved.
        """
        self.params: dict = params

    def log_params(self) -> None:
        """Log Hyperparameters."""
        wandb.init(project=self.params['proj_name'], entity="appl-dl-team-09")
        wandb.config = self.params

    def log_metrics(self, metrics: dict) -> None:
        """Log Metrics.

        :param metrics: Dictionary including description of metrics
            and metric-values.
        """
        assert isinstance(metrics, dict)
        wandb.log(metrics)
