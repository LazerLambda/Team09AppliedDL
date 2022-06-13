"""Test for ImbalancedLoss.py."""

import pytest
import torch

from src import ImbalancedLoss as ImbL

n = 2000


def test_1():
    """Test for Correct Parameters."""
    _ = ImbL.ImbalancedLoss("cpu", 0.2)

    with pytest.raises(Exception):
        ImbL.ImbalancedLoss("cpu", -1, 2)

    with pytest.raises(Exception):
        ImbL.ImbalancedLoss("cpu", -1, 0.5)

    with pytest.raises(Exception):
        ImbL.ImbalancedLoss("cpu", -1, 0.5)

    with pytest.raises(Exception):
        ImbL.ImbalancedLoss(torch.device("cpu"), 0.5, -0.5)

    with pytest.raises(Exception):
        ImbL.ImbalancedLoss(-2, 0.5, -0.5)


def test_2():
    """Check `sigmoid_loss` for correct properties."""
    loss = ImbL.ImbalancedLoss("cpu", 0.5)

    with pytest.raises(Exception):
        loss.sigmoid_loss(torch.rand(20), torch.rand(2))

    with pytest.raises(Exception):
        loss.sigmoid_loss(torch.tensor([]), torch.rand(2))

    with pytest.raises(Exception):
        loss.sigmoid_loss(torch.rand(2), torch.tensor([]))

    rand_loss = loss.sigmoid_loss(
        torch.rand(n), torch.sign(torch.empty(n).random_(2) - 0.5)
    )
    assert torch.prod(rand_loss) >= 0


def test_3():
    """Check `sum_exp` for correct shape of output and type."""
    loss = ImbL.ImbalancedLoss("cpu", 0.5)
    res = loss.sum_exp(torch.rand(n), -1)
    assert len(res.shape) == 0
    assert isinstance(res, torch.Tensor)


def test_4():
    """Test  `imbalanced_nnpu` for correct properties."""
    loss = ImbL.ImbalancedLoss("cpu", 0.5)
    assert loss.imbalanced_nnpu(torch.tensor([]), torch.tensor([])) == 0
    assert loss.imbalanced_nnpu(torch.rand(n), torch.rand(n)) >= 0


def test_5():
    """Test  `nn_balance_pn` for correct properties."""
    loss = ImbL.ImbalancedLoss("cpu", 0.5)
    p_set = torch.tensor(n)
    u_set = torch.tensor(n * 2)
    assert loss.imbalanced_nnpu(torch.tensor([]), torch.tensor(n)) >= 0
    assert loss.imbalanced_nnpu(torch.tensor([]), torch.tensor(n)) >= max(
        0, loss.imbalanced_nnpu(p_set, u_set)
    )


def test_6():
    """Test  `__call__` for correct behavior."""
    loss = ImbL.ImbalancedLoss("cpu", 0.5)
    assert\
        loss(torch.rand(n), torch.sign(torch.empty(n).random_(2) - 0.5)) >= 0
