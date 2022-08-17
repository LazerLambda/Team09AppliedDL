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
    def get_mlp1() -> nn.Module:
        """Return MLP 1.

        :return: MLP model.
        """
        return nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(240, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 1),
        )

    @staticmethod
    def get_transformer() -> nn.Module:
        """Return Transformer.

        :return: Transformer.
        """
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        return nn.Sequential(
            # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
            nn.Flatten(1, -1),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=125,
                    nhead=5,
                    dim_feedforward=2048,
                    activation=nn.functional.relu,
                    layer_norm_eps=1e-5,
                    batch_first=False,
                    norm_first=False
                ),
                num_layers=1,
                norm=None
            ),
            nn.Linear(125, 1),
            nn.Sigmoid()
        )
