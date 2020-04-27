import argparse
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=200, type=int,  help='number of total epochs to run')
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be cifar100')
    parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--teacher', default='', type=str, help='teacher student name')
    parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
    parser.add_argument('--teacher-checkpoint', default='', type=str, help='optional pretrained checkpoint for teacher')
    parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
    parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
    parser.add_argument('--trial-id', default='[trial_id_not_used]', type=str,  help='dataset directory')
    parser.add_argument('--test-run', action='store_true', help='test everything loads correctly')
    parser.add_argument('--teacher-wbits', type=int, default=8)
    parser.add_argument('--teacher-abits', type=int, default=8)
    parser.add_argument('--teacher-quantization', default='dorefa')
    parser.add_argument('--student-wbits', type=int, default=4)
    parser.add_argument('--student-abits', type=int, default=4)
    parser.add_argument('--student-quantization', default='dorefa')
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    """
    Loads weights from checkpoint
    :param model: a pytorch nn student
    :param str checkpoint_path: address/path of a file
    :return: pytorch nn student with weights loaded from checkpoint
    """
    model_ckp = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_ckp['model_state_dict'])
    print(f'after loading model state, is model cuda? {next(model.parameters()).is_cuda}')

    return model
