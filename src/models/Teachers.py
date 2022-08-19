"""Container Class for Teacher-Models."""
from torch import nn


class Teachers:
    """Container Class for Teacher-Models."""

    @staticmethod
    def get_debug_teacher() -> nn.Module:
        """Return Simple Teacher Model for Debugging.

        :return: Simple Model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(240, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def get_lm() -> nn.Module:
        """Return simple Linear Modl.

        :return: Simple Model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def get_mlp1() -> nn.Module:
        """Return MLP 1.

        :return: MLP model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(240, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 1), nn.Sigmoid()
        )
