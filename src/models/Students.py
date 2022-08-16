"""Container Class for Student-Models."""
from torch import nn


class Students:
    """Container Class for Student-Models."""

    @staticmethod
    def get_debug_student() -> nn.Module:
        """Return simple Linear Modl.

        :return: Simple Model.
        """
        return nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )
