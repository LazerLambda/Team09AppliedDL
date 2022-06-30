from models.Teachers import Teachers
from models.Students import Students

from torch.utils import optim

def main():

    teacher: any = Teachers.get_debug_teacher()
    student: any = Students.get_debug_student()

    data_train: any = 

    distil = Distillation(
        student,
        optim.Adam(student.parameters()),
        0.0001,
        teacher,
        optim.Adam(teacher.parameters()),
        0.0001,
        data_train,
        batch_size=2,
        teacher_epochs=2,
        meta_epochs=2,
        alpha=0.5,
        beta=0.5,
        t=3
    )

    distil.train_loop(0.5, 0.5)




if __name__ == "__main__":
    main()