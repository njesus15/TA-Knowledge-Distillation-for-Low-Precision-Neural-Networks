"""
Experimental implementation of Trained Ternary Quantization
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim


from utils.cli_parser import parse_args
from utils.dataset_loader import get_cifar100
from utils.model_manager import get_model
from utils.state_manager import load_checkpoint

# train_loader, test_loader = get_cifar100()

# https://colab.research.google.com/drive/1eEBP46R1QKnKrVJuPCBQT_U-QTAvCIGH

# print(iter(train_loader).next())


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    dataset = 'cifar100'
    num_classes = 100
    # @Brian may change if we use Tiny-Imagenet

    # if train teacher model
    teacher_model = None
    
    if args.teacher:
        teacher_model = get_model(args.teacher, dataset, num_classes, use_cuda=args.cuda)
        if args.teacher_checkpoint:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
        else:
            print("---------- Training Teacher -------")
            train_loader, test_loader = get_cifar100()
            teacher_train_config = copy.deepcopy(train_config)
            teacher_name = '{}_{}_best.pth.tar'.format(args.teacher, trial_id)
            teacher_train_config['name'] = args.teacher
            teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
            teacher_trainer.train()
            teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
        
    
    student_model = get_model(args.student, dataset, num_classes, use_cuda=args.cuda)

    print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    train_loader, test_loader = get_cifar100()
    student_train_config['name'] = args.student
    student_trainer = TrainManager(student_model, teacher=teacher_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
    best_student_acc = student_trainer.train()
    nni.report_final_result(best_student_acc)

