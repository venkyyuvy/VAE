import torch
from torchvision import datasets, transforms

def get_mnist_dataset(batch_size = 64):
    train_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = datasets.MNIST(
        root='./mnist_data/', 
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers= 4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers= 4
    )
    return train_loader, test_loader

def get_cifar_dataset(batch_size = 64):
    train_dataset = datasets.CIFAR10(
        root='./cifar_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = datasets.CIFAR10(
        root='./cifar_data/', 
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers= 4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers= 4
    )
    return train_loader, test_loader

