
from torch import nn


class Students:

    @staticmethod
    def get_debug_student():
        return nn.Sequential(
          nn.Flatten(1, -1),
          nn.Linear(125, 1),
          nn.Sigmoid()
        )

    @staticmethod
    def get_mlp_1():
        return None # TODO