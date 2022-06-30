
from torch import nn

class Teachers:

    @staticmethod
    def get_debug_teacher():
        return nn.Sequential(
          nn.Flatten(1, -1),
          nn.Linear(125, 48),
          nn.ReLU(),
          nn.Linear(48, 1),
          nn.Sigmoid()
        )

    @staticmethod
    def get_transformer():
        return None # TODO