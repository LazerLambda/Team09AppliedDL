"""Class for Loss Function for Distillation."""

from multiprocessing.spawn import import_main_path
import torch
from ImbalancedLoss import ImbalancedLoss


class Distillation_loss:

  def __init__(self, alpha: float) -> None:
        """Initialize Class.

        Formular for loss: L = alpha * L_teacher(cross entropy) + (1- alpha) * L_data_original(Inbalanced loss)
.
        :alpha: Parameter for weighting the two losses
        """
        self.imb_loss: ImbalancedLoss = ImbalancedLoss(
          torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          p=0.5
        )
        self.alpha: float = alpha
        self.cel: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

  def __call__(self, predictions_student_teacher: torch.Tensor, predictions_student_original: torch.Tensor,
               labels_teacher: torch.Tensor, labels_original: torch.Tensor
    ) -> torch.Tensor:
        """Execute on Call.

        # TODO: Rename
        :predictions_student_teacher: predictions of the student made on the teacher dataset.
        :predictions_student_original: predictions of the student made on the original label dataset (data_teacher's complement on training data set).
        :data_teacher: labels from the teacher model on a subset of the dataset.
        :data_original: labels from the original dataset.
        :alpha: Parameter for weighting the two losses
        :return: Loss for batch.
        """
        loss_original = self.imb_loss(predictions_student_original, labels_original)
        loss_teacher = self.cel(predictions_student_teacher, labels_teacher)

        return self.alpha * loss_teacher + (1 - self.alpha) * loss_original