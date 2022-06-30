"""Entry point for Debugging."""
from torch import optim

from data.make_dataset import DebugDataset
from Dataset import AminoDS
from distillation.Distillation import Distillation
from models.Students import Students
from models.Teachers import Teachers


def main():
    """Start program."""
    teacher: any = Teachers.get_debug_teacher()
    student: any = Students.get_debug_student()

    test_data: any = DebugDataset(10, 5)
    test_data.create_debug_dataset()
    data_train: any = AminoDS('TEST_DATA.gzip', True)

    distil = Distillation(
        student,
        optim.Adam(student.parameters()),
        0.0001,
        teacher,
        optim.Adam(teacher.parameters()),
        0.0001,
        data_train,
        batch_size=2,
        student_epochs=2,
        teacher_epochs=2,
        meta_epochs=2,
        alpha=0.5,
        beta=0.5,
        t=3
    )

    distil.train_loop(0.5, 0.5)

    test_data.rm_csv()


if __name__ == "__main__":
    main()
