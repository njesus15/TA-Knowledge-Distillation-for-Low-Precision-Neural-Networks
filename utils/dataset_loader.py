from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_cifar100():
    pin_memory = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=pin_memory
    )

    testset = CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=pin_memory)
    
    return train_loader, test_loader