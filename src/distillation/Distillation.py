"""Module for Distillation."""

import torch
from ignite.contrib.metrics import ROC_AUC
from ignite.engine.engine import Engine
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from distillation.Train import Train
from Logger import Logger
from loss.DistillationLoss import DistillationLoss
from loss.ImbalancedLoss import ImbalancedLoss


class Distillation:
    """Distillation Class.

    Distill a (smaller) student model from a teacher model via
    online learning. Both models are trained simultaneuosly.
    The teacher model is trained using a specific loss function
    only depending on the provided training dataset. Furthermore,
    the teacher's softmax function is softened using a higher
    temperature (usually 2.5-4) to avoid peaks in the softmax
    distribution. After some epochs of training (`teacher_epochs`)
    the student model is trained by using a dedicated student model
    (`student_epochs` times) loss function, which consists of a loss
    function aimed at learning the teacher's learned distribution
    and the same loss function as the teacher with temperature set
    to 1.(TODO: ????)

    :method train_loop: Train models.

    """

    def __init__(
            self,
            student: nn.Module,
            student_optim: optim.Optimizer,
            student_lr: float,
            teacher: nn.Module,
            teacher_optim: optim.Optimizer,
            teacher_lr: float,
            data_train: Dataset,
            data_test: Dataset,
            batch_size: int,
            student_epochs: int,
            teacher_epochs: int,
            meta_epochs: int,
            alpha: float,
            beta: float,
            t: float,
            logger: Logger = None,
            device: torch.device = None,
            *args,
            **kwargs):
        """Initialize Distillation Class.

        # TODO Write doc
        :param student: nn.Module,
        :param student_optim: optim.Optimizer,
        :param student_lr: float,
        :param teacher: nn.Module,
        :param teacher_optim: optim.Optimizer,
        :param teacher_lr: float,
        :param data_train: Train-Dataset,
        :param data_test: Test-Dataset,
        :param batch_size: int,
        :param student_epochs: int,
        :param teacher_epochs: int,
        :param meta_epochs: int,
        :param alpha: float,
        :param beta: float,
        :param t: float,
        :param logger: Logger class for MLflow.
        :param device: Device training should be computed on.
        :param *args: Additional params.
        :param **kwargs:  Additional params.
        """
        check_cond = lambda e: e >= 0 and e <= 1
        assert isinstance(student, nn.Module),\
            "ERROR: `student` must be of class nn.Module."
        assert check_cond(student_lr),\
            "ERROR: `student_lr` must be in domain of [0,1]."
        # assert isinstance(student_optim, optim.Optimizer),\
        #     "ERROR: `student_optim` must be of class optim.Optimizer."
        assert isinstance(teacher, nn.Module),\
            "ERROR: `teacher` must be of class nn.Module."
        assert check_cond(teacher_lr),\
            "ERROR: `teacher_lr` must be in domain of [0,1]."
        # assert isinstance(teacher_optim, optim.Optimizer),\
        #     "ERROR: `teacher_optim` must be of class optim.Optimizer."
        assert isinstance(data_train, torch.utils.data.Dataset),\
            "ERROR: `data_train` must be of class utils.data.Dataset."
        assert isinstance(data_test, torch.utils.data.Dataset),\
            "ERROR: `data_test` must be of class utils.data.Dataset."
        assert batch_size > 0,\
            "ERROR: `batch_size` must be larger than 0."
        assert teacher_epochs > 0,\
            "ERROR: `teacher_epochs` must be larger than 0."
        assert meta_epochs > 0,\
            "ERROR: `meta_epochs` must be larger than 0."
        assert check_cond(alpha),\
            "ERROR: `alpha` must be in domain of [0,1]."
        assert check_cond(beta),\
            "ERROR: `beta` must be in domain of [0,1]."
        assert t > 0,\
            "ERROR: `t` must be larger than 0."
        # TODO check device
        assert isinstance(logger, Logger),\
            "ERROR: `logger` must be of class Logger."

        self.student: nn.Module = student
        self.teacher: nn.Module = teacher
        self.data_train: Dataset = data_train
        self.data_test: Dataset = data_test
        self.batch_size: int = batch_size
        # self.teacher_epochs: int = teacher_epochs
        self.meta_epochs: int = meta_epochs
        self.alpha: float = alpha
        self.beta: float = beta
        self.t: float = t
        self.logger: Logger = logger

        if device is None:
            self.device: torch.Device =\
                torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")

        # TODO rm device here
        imb_loss: ImbalancedLoss = ImbalancedLoss(
            torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"), p=0.5)
        dist_loss: DistillationLoss =\
            DistillationLoss(alpha=0.5)
        self.trainer_teacher: Train = Train(
            teacher,
            teacher_optim,
            batch_size,
            teacher_epochs,
            imb_loss,
            teacher_lr,
            self.device)
        self.trainer_student: Train = Train(
            student,
            student_optim,
            batch_size,
            student_epochs,
            dist_loss,
            student_lr,
            self.device)

    def eval_models(self, data: Dataset, desc: str = "") -> tuple:
        """Evaluate Student and Teacher Model.

        :param data: Dataset model will be evaluated on.
        :param desc: Description string for tqdm.

        :return: Tuple with metrics for teacher and student (teacher first).
        """
        self.teacher.eval()
        self.student.eval()

        eval_collector_teacher_pred: list = []
        eval_collector_teacher_labl: list = []
        eval_collector_student_pred: list = []
        eval_collector_student_labl: list = []

        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        for _, [x, y] in enumerate(tqdm(dataloader, desc=f'Evaluate-{desc}')):
            x = x.to(self.device).float()

            with torch.no_grad():
                pred_teacher_tmp = self.teacher(x)
                pred_student_tmp = self.student(x)

                eval_collector_teacher_pred.append(pred_teacher_tmp.cpu())
                eval_collector_teacher_labl.append(y)
                eval_collector_student_pred.append(pred_student_tmp.cpu())
                eval_collector_student_labl.append(y)

        self.teacher.train()
        self.student.train()

        pred_teacher: torch.Tensor = torch.concat(eval_collector_teacher_pred)
        pred_student: torch.Tensor = torch.concat(eval_collector_student_pred)

        labl_teacher: torch.Tensor = torch.concat(eval_collector_teacher_labl)
        labl_student: torch.Tensor = torch.concat(eval_collector_student_labl)

        return self.eval(
            pred_teacher,
            labl_teacher,
            pred_student,
            labl_student)

    # TODO: @staticmethod ?
    def eval(
            self,
            y_teacher: torch.Tensor,
            labl_teacher: torch.Tensor,
            y_student: torch.Tensor,
            labl_student: torch.Tensor) -> tuple:
        """Evaluate Predicted Labels.

        :param y_teacher: True labels for teacher.
        :param labl_teacher: Predicted labels for teacher.
        :param y_student: True labels for student.
        :param labl_student: Predicted labels for student.

        :return: AUC for teacher and student (teacher first).
        """
        default_evaluator = Engine(lambda _, batch: batch)
        roc_auc = ROC_AUC()
        roc_auc.attach(default_evaluator, 'roc_auc')

        state_teacher = default_evaluator.run([[y_teacher, labl_teacher]])
        state_student = default_evaluator.run([[y_student, labl_student]])

        return state_teacher.metrics['roc_auc'],\
            state_student.metrics['roc_auc']

    def print_table(
            self,
            meta_epoch: int,
            auc_teacher_train: float,
            auc_teacher_test: float,
            auc_student_train: float,
            auc_student_test: float) -> None:
        """Print Results.

        :param meta_epoch: Meta-epoch.
        :param auc_teacher_train: AUC for teacher on training data.
        :param auc_teacher_test: AUC for teacher on test data.
        :param auc_student_train: AUC for student on training data.
        :param auc_student_test: AUC for student on test data.
        """
        info_line: str = (
            f"Meta Epoch: {meta_epoch:.3f}\t "
            f"AUC Tchr. Trn.: {auc_teacher_train:.3f}\t "
            f"AUC Tchr. Tst.: {auc_teacher_test:.3f}\t "
            f"AUC Stdnt. Trn.: {auc_student_train:.3f}\t "
            f"AUC Stdnt. Tst.: {auc_student_test:.3f}"
        )
        print(info_line)

    def train_loop(self, alpha, beta) -> None:
        """Train Teacher and Student.

        :param alpha: Alpha parameter for distillation-loss function.
        :param beta: Parametr for transfer-data-original-data split.
        """
        auc_list: list = []
        for meta_epoch in tqdm(
                range(self.meta_epochs), desc='Meta-Epoch'):

            self.trainer_teacher.train_teacher(self.data_train)

            self.trainer_student.train_student(
                self.data_train,
                self.teacher,
                alpha,
                beta)

            auc_student_train, auc_teacher_train = self.eval_models(
                self.data_train, desc="Teacher")
            auc_student_test, auc_teacher_test = self.eval_models(
                self.data_test, desc="Student")

            auc_list.append((
                meta_epoch,
                auc_teacher_train,
                auc_teacher_test,
                auc_student_train,
                auc_student_test))

            self.logger.log_metrics({
                "AUC Tchr. Trn.": auc_teacher_train,
                "AUC Tchr. Tst.": auc_teacher_test,
                "AUC Stdnt. Trn.": auc_student_train,
                "AUC Stdnt. Tst.": auc_student_test
            })

        for e in auc_list:
            self.print_table(e[0], e[1], e[2], e[3], e[4])
