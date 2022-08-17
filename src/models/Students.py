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
                    d_model=240,
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
            nn.Linear(240, 1),
            nn.Sigmoid()
        )
