import os
import torch
import torchvision.datasets as dset 
import dataloaders.data_utils as utils
from .hdf5 import H5Dataset

def define_dataloader(dataset='CIFAR10', data='data', batch_size=64, download=False):
    if dataset == 'CIFAR10':
        train_transform, valid_transform = utils._data_transforms_cifar10()
        train_data = dset.CIFAR10(root=data, train=True, download=download, transform=train_transform)
        valid_data = dset.CIFAR10(root=data, train=False, download=download, transform=valid_transform)
    elif dataset == 'CIFAR100':
        train_transform, valid_transform = utils._data_transforms_cifar100()
        train_data = dset.CIFAR100(root=data, train=True, download=download, transform=train_transform)
        valid_data = dset.CIFAR100(root=data, train=False, download=download, transform=valid_transform)
    elif dataset == 'SVHN':
        train_transform, valid_transform = utils._data_transforms_svhn()
        train_data = dset.SVHN(root=data, split='train', download=download, transform=train_transform)
        valid_data = dset.SVHN(root=data, split='test', download=download, transform=valid_transform)
    elif dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from .DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700
    elif dataset == 'imagenet-1k':
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
        ])

        train_data = H5Dataset(os.path.join(data, 'imagenet-train-256.h5'), transform=train_transform)



    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

