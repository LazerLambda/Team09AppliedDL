"""Container Class for Teacher-Models."""
from torch import nn


class Teachers:
    """Container Class for Teacher-Models."""

    @staticmethod
    def get_debug_teacher():
        """Return Simple Teacher Model for Debugging.

        :return: Simple Model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(125, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def get_transformer():
        """Return Transformer.

        :return: Transformer.
        """
        return None  # TODO
