import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False

def _get_parser():
    # parser from https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=200, type=int,  help='number of total epochs to run')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--teacher', default='', type=str, help='teacher student name')
    parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
    parser.add_argument('--teacher-checkpoint', default='', type=str, help='optinal pretrained checkpoint for teacher')
    parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
    parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
    return parser

def parse_args():
    parser = _get_parser()
    args = parser.parse_args()
    return args

'''
# example tests

cmd = 'python3 train.py --epochs 160 --teacher resnet110 --student resnet8 --cuda 1 --dataset cifar10'
parser.parse_args(cmd.split(' ')[2:])
'''