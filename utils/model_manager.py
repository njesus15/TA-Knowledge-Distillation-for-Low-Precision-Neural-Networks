import sys
sys.path.append('..') 
from resnet_cifar import *

def get_model(name, dataset="cifar100", num_classes=100, use_cuda=False):
    assert name[:6] == 'resnet'
    resnet_size = name[6:]
    resnet_model = resnet_models.get(resnet_size)(num_classes=num_classes)
    model = resnet_model

	# copy to cuda if activated
    if use_cuda:
        model = model.cuda()
    
    return model