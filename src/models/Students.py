"""Container Class for Student-Models."""
from torch import nn


class Students:
    """Container Class for Student-Models."""

    @staticmethod
    def get_debug_student():
        """Return simple Linear Modl.

        :return: Simple Model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(125, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def get_mlp_1():
        """Return MLP 1.

        :return: Simple MLP 1.
        """
        return None  # TODO
