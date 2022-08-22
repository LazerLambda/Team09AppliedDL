Group09
==============================

# Sequence-level Knowledge Distillation

---

## Introduction

In this Git repository a new way of knowlegde distillation for data with sequence-level input and binary output on an inbalanced dataset was implemented. This work was inspired by the papers [1] and [2].

## Concept 

The idea behind this particullar distillation approach is for the student model to learn the distribution of the teacher model whilst learning from the original data itself. This is done using an online learning approach (teacher and student are trained simultaneously).


## Methodology

In order for the student model to learn from the teacher model a combined loss for training was used. The first part of the loss contains the original data and the loss for unlabled positive data (implemented in the loss class, the algorithm is discribed in [3]). The second part is a cross-entropy loss on data labled by the teacher model. These two losses are combined in a convex combination with the hyperparameter $\alpha \in (0,1)$. <br>

With the above discribed loss the student model is trained. The teacher model is trained using the loss for unlabled positive data. 

## Algorithm

**Input:**  <br>
           training data (subset of original data); <br>
           hyperparameters for loss: $\alpha \in (0,1)$, $\beta \in (0,1)$; <br>
           hyperparameters for epochs: meta_epoch, teacher_epoch, student_epoch; <br>
           models: student model **S**, teacher model **T** (both untrained) <br>
           
**Output:**  <br>
           trained models **S** and **T**

1. **For** each meta_epoch **do**:
2. > **For** each teacher_epoch **do**: 
3. >> Train teacher model with training data
3. > **For** each student_epoch **do**:
4. >> Shuffle data and take a batch for training iteration.
5. >> Split batch into two disjoint data sets $data_s$ and $data_t$ with $n_{data_s} = \beta * n_{data}$ and $n_{data_t} = (1-\beta) * n_{data}$
6. >> Make predictions with **T** for $data_t$
7. >> Train **S** with both data sets (use predictions from **T** for $data_t$ and true labels for $data_s$) using a combined loss weighted with $\alpha$ for $L_t$ and $(1- \alpha)$ for $L_s$
8. Save **S** and **T**


## Code Structure

In the folder src one can find the folders distillation, loss, models and visualization. <br>
All losses are implemented in the folder loss. <br>
The training and the distillation algorithm is located in the folder distillation. <br>
The models used for this work can be found in the folder models. <br>
The Config file (hyperparameters.yml) can be found in Config. <br>
Results and graphics can be found in the Wiki part of this Github repository.

## Reproduce our results

In order to reproduce our results, adjust the file run_file_google_colab.py in the folder 'notebooks' with your own data path and git key, and run it on GoogleColab. If you are using another device than GoogleColab, please execute:
1. > `!python3 -m pip install -r requirements.txt`
2. > \# Adjust your path to data (e.g. connect to google drive)
2. > `os.chdir('./src')`
3. > `from main import main`
4. > `
        main({
            'config_path' :'/Team09AppliedDL/config/hyperparameters.yml',
            'data_path' : 'path to data',
            'wandb' : True})
     `

## References
[1] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, 2015. *Distilling the Knowledge in a Neural Network*. https://arxiv.org/abs/1503.02531 <br>
[2] Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao, 2021. *Knowledge Distillation: A Survey*. https://arxiv.org/abs/2006.05525v7 <br>
[3] Guangxin Su, Weitong Chen, Miao Xu, 2021. *Positive-Unlabeled Learning from Imbalanced Data*. https://www.ijcai.org/proceedings/2021/412 <br>

Project Organization
------------

    ├── config 
    │   ├── hyperparameters.yml  <- YML-File for hyperparameters and model specification.
    │    
    ├── notebooks          <- Jupyter notebooks.
    │   ├── run_file_google_colab.ipynb  <- Notebook for running the code on GoogleColab
    │
    ├── reports            <- folder with images containing reported final results           
    │   ├── figures
    │   │   ├── auc_student_test.png
    │   │   ├── auc_student_train.png
    │   │   ├── auc_teacher_test.png
    │   │   ├── auc_teacher_test.png
    │
    ├── src                <- Source code to use in this project
    │   │
    │   ├── data           <- Scripts to preprocess data
    │   │   ├── Dataset.py        <- Script for data preparation (read in and one hot encoding of the original dataset) + preparation of data for DNABert
    │   │   ├── make_dataset.py   <- Script for generating random test data
    │   │
    │   ├── distillation          <- Script for the distillation class with evaluation and train loop
    │   │   ├── Distillation.py
    │   │   ├── Train.py
    │
    │   ├── loss  
    │   │   ├── DistillationLoss.py  <- Script for our distillation loss
    │   │   ├── ImbalancedLoss.py    <- Script for the imbalanced loss
    │
    │   ├── models             <- Scripts for teacher and student models
    │   │   ├── Students.py    <- transformer and mlp's
    │   │   ├── Teachers.py    <- mlp's
    │
    │   ├── ConfigReader.py    <- Script to read configurations
    │   ├── Logger.py          <- Script for including mlflow and wandb
    │   ├── main.py            <- Launch script
    │
    ├── tests
    │   ├── DebugDataset.py          <- Script for generating random test data
    │   ├── test_Dataset.py          <- EMPTY
    │   ├── test_ImbalancedLoss.py   <- testing the imbalanced loss for correct parameters, properties, type and shape of outouts and for correct behaviour
    │   
    ├── LICENSE
    ├── Makefile               <- Makefile with commands like `make data` or `make train`
    ├── README.md              <- The top-level README for developers using this project.
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment and installing all required packages
    ├── setup.py               <- makes project pip installable (pip install -e .) so src can be imported
    ├── test_environment.py    <- Test for correct python version
    ├── tox.ini                <- Tox file. Run for tests and linting
   
 

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
