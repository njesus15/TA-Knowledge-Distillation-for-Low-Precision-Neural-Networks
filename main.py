import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.config_manager import get_config, get_train_config
from utils.train_manager import TrainManager, train_student, train_teacher
from utils.setup_manager import parse_arguments, load_checkpoint
from resnet import get_model, is_resnet

if __name__ == "__main__":
    # Parsing arguments and prepare settings for training
    args = parse_arguments()
    print(args)
    config = get_config()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    train_config = get_train_config(args, config)

    if args.test_run:
        print('test run succeeded')
        exit()
    # train teacher
    teacher_model = None
    if args.teacher:
        teacher_model = train_teacher(args, train_config)

    # train student
    train_student(args, train_config, teacher_model)
