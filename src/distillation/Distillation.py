import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm

from loss.ImbalancedLoss import ImbalancedLoss
from loss.DistillationLoss import Distillation_loss
from Train import Train

class Distillation:
    """Distillation Class.
    
    Distill a (smaller) student model from a teacher model via online learning.
    Both models are trained simultaneuosly. The teacher model is trained using 
    a specific loss function only depending on the provided training dataset.
    Furthermore, the teacher's softmax function is softened using a higher temperature
    (usually 2.5-4) to avoid peaks in the softmax distribution. After some epochs
    of training (`teacher_epochs`) the student model is trained by using a dedicated
    student model(`student_epochs` times) loss function, which consists of a loss function aimed
    at learning the teacher's learned distribution and the same loss function as the teacher
    with temperature set to 1.(TODO: ????)

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
            data: Dataset,
            batch_size: int,
            student_epochs: int,
            teacher_epochs: int,
            meta_epochs: int,
            alpha: float,
            beta: float,
            t: float,
            *args,
            **kwargs):
        """Initialize Distillation Class.
      
        """
        check_cond = lambda e : e >= 0 and e <= 1
        assert isinstance(student, nn.Module), "ERROR: `student` must be of class nn.Module."
        assert check_cond(student_lr), "ERROR: `student_lr` must be in domain of [0,1]."
        # assert isinstance(student_optim, optim.Optimizer), "ERROR: `student_optim` must be of class optim.Optimizer."
        assert isinstance(teacher, nn.Module), "ERROR: `teacher` must be of class nn.Module."
        assert check_cond(teacher_lr), "ERROR: `teacher_lr` must be in domain of [0,1]."
        # assert isinstance(teacher_optim, optim.Optimizer), "ERROR: `teacher_optim` must be of class optim.Optimizer."
        assert isinstance(data, torch.utils.data.Dataset), "ERROR: `data` must be of class utils.data.Dataset."
        assert batch_size > 0, "ERROR: `batch_size` must be larger than 0."
        assert teacher_epochs > 0, "ERROR: `teacher_epochs` must be larger than 0."
        assert meta_epochs > 0, "ERROR: `meta_epochs` must be larger than 0."
        assert check_cond(alpha), "ERROR: `alpha` must be in domain of [0,1]."
        assert check_cond(beta), "ERROR: `beta` must be in domain of [0,1]."
        assert t > 0, "ERROR: `t` must be larger than 0."

        self.student: nn.Module = student
        self.teacher: nn.Module = teacher
        self.data: any = data # TODO
        # self.teacher_epochs: int = teacher_epochs
        self.meta_epochs: int = meta_epochs
        self.alpha: float = alpha
        self.beta: float = beta
        self.t: float = t

        # TODO rm device here
        imb_loss: ImbalancedLoss = ImbalancedLoss(torch.device("cuda" if torch.cuda.is_available() else "cpu"), p=0.5)
        dist_loss: Distillation_loss = Distillation_loss(alpha=0.5) 
        self.trainer_teacher: Train = Train(
            teacher,
            teacher_optim,
            batch_size,
            teacher_epochs,
            imb_loss,
            teacher_lr) 
        self.trainer_student: Train = Train(
            student,
            student_optim,
            batch_size,
            student_epochs,
            dist_loss,
            student_lr)

    def train_loop(self, alpha, beta):

        for _ in tqdm(
                range(self.meta_epochs), desc='Meta-Epoch'):

            self.trainer_teacher.train_teacher(self.data)

            self.trainer_student.train_student(self.data, self.teacher, alpha, beta)
