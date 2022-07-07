"""Logger Class."""

import mlflow


class Logger:
    """Logger Class.
    
    :method __init__: Initialize class.
    :method log_params: Log hyperparameters.
    :method log_metrics: Log metrics.
    """
    def __init__(self, params: dict):
        """Initialize Class.

        :params: Dictionary for hyperparameters to be saved.
        """
        self.params: dict = params

    def log_params(self) -> None:
        """Log Hyperparameters"""
        for key, value in self.params.items():
            mlflow.log_param(key, value)
            
    def log_metrics(self, metrics: dict) -> None:
        """Log Metrics.

        :param: Dictionary including description of metrics
            and metric-values.
        """
        for key, value in self.params.items():
            mlflow.log_metrics(key, value)