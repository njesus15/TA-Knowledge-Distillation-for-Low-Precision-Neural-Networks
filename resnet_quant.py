"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math
from quantization.q_funcs import *
from utils.dataset_loader import get_dataset

class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)
        # @BL fix for now
        self.weights_q = self.qfn(self.weight)
        self.weights_fp = self.weight


    def forward(self, x):
        
        out = nn.functional.conv2d(x, self.weights_q, self.bias, self.stride,
                                    self.padding)
        return out

'''
# @BL bug with setting weights_q and weights_fp when loading
class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.qfn = weight_quantize_fn(w_bit=wbit)


    def forward(self, x):
        weights_q = self.qfn(self.weight)
        self.weights_q = weights_q
        self.weights_fp = self.weight
        return nn.functional.conv2d(x, weights_q, self.bias, self.stride,
                                    self.padding)
'''


class PreActBasicBlock_convQ(nn.Module):
    expansion = 1

    def __init__(self, q_method, wbit, abit, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBasicBlock_convQ, self).__init__()
        self.act_qfn = activation_quantize_fn(a_bit=abit)
        # TODO: Move class from this file
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2d_Q(wbit, in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False, q_method=q_method)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = Conv2d_Q(wbit, out_planes, out_planes, stride=1, kernel_size=3, padding=1, q_method=q_method)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        # activation qunatization applied here
        out = self.bn1(x)
        out = self.act_qfn(self.relu(out))

        # TODO: check how residual is accounted for. DoReFa seems to leave residual full precision??
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act_qfn(self.relu(out))
        out = self.conv2(out)

        out += residual

        return out



class PreActBottleneck_Q(nn.Module):
    expansion = 4

    def __init__(self, q_method, wbit, abit, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck_Q, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d_Q(wbit, inplanes, planes, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_Q(wbit, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_Q(wbit, planes, planes * 4, kernel_size=1, stride=stride, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

        self.act_qfn = activation_quantize_fn(a_bit=abit)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act_qfn(self.relu(out))

        # TODO: check if quant is needed here
        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act_qfn(self.relu(out))

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act_qfn(self.relu(out))

        out = self.conv3(out)
        # TODO: Again, check if residual needs and qunatization
        out += residual

        return out




class PreAct_ResNet_Cifar_Q(nn.Module):

    def __init__(self, block, layers, wbit, abit, num_classes=10, q_method='dorefa'):
        super(PreAct_ResNet_Cifar_Q, self).__init__()
        self.inplanes = 16
        # first conv is not quantized
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0], wbit, abit, q_method=q_method)
        self.layer2 = self._make_layer(block, 32, layers[1], wbit, abit, stride=2, q_method=q_method)
        self.layer3 = self._make_layer(block, 64, layers[2], wbit, abit, stride=2, q_method=q_method)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # TODO: There is no linear quantization done here... check paper to see what they discuss
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, wbit, abit, stride=1, q_method='dorefa'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(q_method, wbit, abit, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(q_method, wbit, abit, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [1, 1, 1], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet14_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [2, 2, 2], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet20_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [3, 3, 3], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet26_cifar(wbit, abit, q_method=None, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [4, 4, 4], wbit, abit, q_method=q_method, **kwargs)
    return model


resnet_models = {
    '8': resnet8_cifar,
    '14': resnet14_cifar,
    '20': resnet20_cifar,
    '26': resnet26_cifar,
}


def is_resnet(name):
    """
    Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
    :param name:
    :return:
    """
    name = name.lower()
    return name.startswith('resnet')


def get_quant_model(name, qparams, dataset="cifar100", use_cuda=False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    num_classes = 100 if dataset == 'cifar100' else 10
    wbits, abits, q_method = qparams

    model = None
    if is_resnet(name):
        resnet_size = name[6:]
        resnet_model = resnet_models.get(resnet_size)
        model = resnet_model(wbits, abits, q_method, num_classes=num_classes)
    else:
        raise Exception('not resnet!')

    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()

    return model
