"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.6
"""
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# standard cast into Tensor and pixel values normalization in [-1, 1] range
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# extra transfrom for the training data, in order to achieve better performance
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
    torchvision.transforms.RandomHorizontalFlip(), 
])

def download_train_data(train: bool)-> torchvision.datasets.cifar.CIFAR10:
    trainset = torchvision.datasets.CIFAR10(
        root='./data/01_raw/', train=train, download=True, transform=train_transform
    )
    return trainset

def download_test_data(train: bool) -> torchvision.datasets.cifar.CIFAR10:
    testset = torchvision.datasets.CIFAR10(
        root='./data/01_raw/', train=train, download=True, transform=transform
    )
    return testset
