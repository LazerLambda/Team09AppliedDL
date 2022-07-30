"""Class for Loss Function for Distillation."""

import torch

from loss.ImbalancedLoss import ImbalancedLoss


class DistillationLoss:
    """Distillation Loss Class.

    Detailed description in __call__ method.
    """

    def __init__(self, alpha: float) -> None:
        """Initialize Class.

        Formular for loss: L = alpha * L_teacher(cross entropy) +
        (1- alpha) * L_data_original(Inbalanced loss).
        :param alpha: Parameter for weighting the two losses
        """
        self.imb_loss: ImbalancedLoss = ImbalancedLoss(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            p=0.5)
        self.alpha: float = alpha
        self.cel: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    def __call__(
            self,
            predictions_student_teacher: torch.Tensor,
            predictions_student_original: torch.Tensor,
            labels_teacher: torch.Tensor,
            labels_original: torch.Tensor) -> torch.Tensor:
        """Execute on Call.

        Compute loss for student. Predictions for transfer
        dataset (obtained by teacher) is weighted with hyperparameter
        alpha. Original is weighted with 1- alpha.

        :param predictions_student_teacher: predictions of the
            student made on the teacher dataset.
        :param predictions_student_original: predictions of the
            student made on the original label dataset (data_teacher's
            complement on training data set).
        :param labels_teacher: labels from the teacher model on a
            subset of the dataset.
        :param labels_original: labels from the original dataset.
        :return: Loss for batch.
        """
        loss_original = self.imb_loss(
            predictions_student_original,
            labels_original)
        loss_teacher = self.cel(
            predictions_student_teacher,
            labels_teacher.long())

        return self.alpha * loss_teacher + (1 - self.alpha) * loss_original
