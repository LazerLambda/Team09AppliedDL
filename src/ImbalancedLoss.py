from typing import Any, Tuple, Union

import torch
from numpy import isin


class ImbalancedLoss:
    """Loss Function for"""

    def __init__(
        self, device: Union[str, torch.device], p: float, p_: float = 0.5
    ) -> None:
        """Initialize Class.

        :param device: Device each tensor will be moved to.
        :param p: TODO
        :param p_: TODO
        """
        assert isinstance(p, float)
        assert isinstance(p_, float)
        assert isinstance(device, torch.device) or isinstance(device, str)
        assert p >= 0 and p <= 1
        assert p_ >= 0 and p_ <= 1

        self.p: float = p
        self.p_: float = p_
        self.device: Union[str, torch.device] = device

    def sum_exp(self, x: torch.Tensor, sgn: int) -> torch.Tensor:
        """Compute Sum over Sigmoid Loss Tensor.

        :param device: Device to compute on.

        :return: Sum of tensor.
        """
        return torch.sum(
            self.sigmoid_loss(
                output=x,
                label=(
                    sgn
                    * torch.ones((x.shape[0] if len(x.shape) > 0 else 0)).to(
                        self.device
                    )
                ),
            )
        )  # TODO: Remove?

    @staticmethod
    def sigmoid_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute Sigmoid Loss Function.

        Compute:
        :param output: predicted value.
        :param device: Device to compute on.

        :return: loss
        """
        product: torch.Tensor = torch.mul(output, label)
        denum: torch.Tensor = torch.add(1, torch.exp(product))
        return torch.div(1, denum)

    def imbalanced_nnpu(
        self, pred_p: torch.Tensor, pred_u: torch.Tensor
    ) -> torch.Tensor:
        """Compute ImbalancednnPU-Loss

        Implementation according to https://www.ijcai.org/proceedings/2021/0412.pdf

        :param pred_p: Positive labeled data.
        :param pred_u: Unlabeled data.
        :param p: Proportion of labeled to unlabeled data.
        :param p_: Proportion of upsampled data.
        :param device: Device to compute on.

        :returns: Loss for batch.
        """
        np: int = pred_p.shape[0] if len(pred_p.shape) > 0 else 0
        cp: float = (1 - self.p_) * self.p / (np * (1 - self.p)) if np != 0 else 0

        nu: int = pred_u.shape[0] if len(pred_u.shape) > 0 else 0
        cu: float = (1 - self.p_) / (nu * (1 - self.p)) if nu != 0 else 0

        sum1: torch.Tensor = self.sum_exp(pred_u, -1)
        sum2: torch.Tensor = self.sum_exp(pred_p, -1)

        return cu * sum1 - cp * sum2

    def nn_balancePN(self, pred_p: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
        """Compute nnBalancePN.

        Implementation according to https://arxiv.org/abs/1703.00593.

        :param pred_p: Positive labeled data.
        :param pred_u: Unlabeled data.
        :param p: Proportion of labeled to unlabeled data.
        :param p_: Proportion of upsampled data.
        :param device: Device to compute on.

        :returns: Loss for batch.
        """
        clipped_imbnnPU: torch.Tensor = torch.max(
            torch.Tensor([0]).to(self.device),
            self.imbalanced_nnpu(pred_p=pred_p, pred_u=pred_u),
        )
        np: int = pred_p.shape[0]
        if np == 0:
            return clipped_imbnnPU
        else:
            return self.p_ / np * self.sum_exp(pred_p, 1) + clipped_imbnnPU

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Execute on Call.

        :param predictions: Tensor of predictions from model.
        :param labels: Tensor of ground truth labels.

        :return: Loss for batch.
        """
        pred_p = predictions[labels == 1].to(self.device)
        pred_u = predictions[labels != 1].to(self.device)

        if self.imbalanced_nnpu(pred_p, pred_u) >= 0:
            # TODO: Sign correct?
            return self.nn_balancePN(pred_p, pred_u)
        else:
            return -self.imbalanced_nnpu(pred_p, pred_u)
