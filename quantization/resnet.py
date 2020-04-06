"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import math
from quantization.q_funcs import get_q_functions


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Conv2d_Q(nn.Conv2d):

    def __init__(self, wbit, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, q_method='dorefa', bias=False):

        super(Conv2d_Q, self).__init__(in_planes, out_planes, kernel_size, stride,
                                       padding, bias=False)
        self.wbit = wbit
        self.qfn, _ = get_q_functions(wbit=wbit, abit=1, method=q_method)

    def forward(self, x):
        weights_q = self.qfn(self.weight)
        return nn.functional.conv2d(x, weights_q, self.bias, self.stride,
                                    self.padding)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBasicBlock_convQ(nn.Module):
    expansion = 1

    def __init__(self, q_method, wbit, abit, in_planes, out_planes, stride, downsample=None):
        super(PreActBasicBlock_convQ, self).__init__()
        _, self.act_qfn = get_q_functions(wbit=0, abit=abit, method=q_method)
        # TODO: Move class from this file
        self.bn1 = nn.BatchNorm2d(in_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d_Q(wbit, in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False, q_method=q_method)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = Conv2d_Q(wbit, out_planes, out_planes, stride=1, kernel_size=3, padding=1, q_method=q_method)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # activation qunatization applied here
        out = self.self.bn1(x)
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


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

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

        _, self.act_qfn = get_q_functions(wbit=1, abit=abit, method=q_method)

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



class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreAct_ResNet_Cifar_Q(nn.Module):

    def __init__(self, block, layers, wbit, abit, num_classes=10, q_method='dorefa'):
        super(PreAct_ResNet_Cifar_Q, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0], wbit, abit, q_method=q_method)
        self.layer2 = self._make_layer(block, 32, layers[1], wbit, abit, stride=2, q_method=q_method)
        self.layer3 = self._make_layer(block, 64, layers[2], wbit, abit, stride=2, q_method=q_method)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # TODO: There is no linear quantization done here... check papaer to see what they discuss
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


def resnet8_cifar(wbit, abit, q_method, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [1, 1, 1], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet14_cifar(wbit, abit, q_method, **kwargs):
    model = PreAct_ResNet_Cifar_Q(PreActBasicBlock_convQ, [2, 2, 2], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet20_cifar(wbit, abit, q_method, **kwargs):
    model = PreAct_ResNet_Cifar_Q(wbit, abit, q_method, PreActBasicBlock_convQ, [3, 3, 3], wbit, abit, q_method=q_method, **kwargs)
    return model


def resnet26_cifar(**kwargs):
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


def get_model(name, qparams, dataset="cifar100", use_cuda=False):
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

if __name__ == '__main__':
    qparams = [8, 8, 'dorefa']
    model = get_model('resnet8', qparams=qparams)
