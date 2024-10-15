import os
import re
import torch
import torchvision
import random
import numpy as np

from tqdm import tqdm
from torchvision import datasets, transforms

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
MNIST_PATH = '{}/datasets/mnist'.format(os.path.join(FILE_PATH, ".."))
CIFAR10_PATH = '{}/datasets/cifar10'.format(os.path.join(FILE_PATH, ".."))
IMAGENET_PATH = '{}/datasets/ImageNet-val'.format(os.path.join(FILE_PATH, ".."))

def get_data_by_id(dataset_name, use_train_data=False, data_path=None):
    cifar10_path = CIFAR10_PATH
    mnist_path = MNIST_PATH
    imagenet_path = IMAGENET_PATH
    # flickr_path =Flickr_PATH
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = None
        test_set = datasets.CIFAR10(
            root=cifar10_path if data_path is None else data_path,
            train=use_train_data, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.Resize(232),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        test_set = datasets.ImageNet(
            root=imagenet_path if data_path is None else data_path,
            split='val' if not use_train_data else 'train', transform=transform)
    elif dataset_name == 'mnist':
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = None
        test_set = datasets.MNIST(
            root=mnist_path if data_path is None else data_path, train=use_train_data, download=True,
            transform=transform)
    else:
        raise NotImplementedError("Dataset not supported")

    return test_set
