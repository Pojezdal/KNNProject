import torch
import torchvision
import os
from enum import Enum
import random
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

class DataLoader:
    def __init__(self, name, dataset, ratio = [0.75, 0.2, 0.05], batch_size = 64):
        self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * ratio) for ratio in ratio])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        print(f"Train size: {len(train_data)}")

cifar10_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
stl10_dataset = torchvision.datasets.STL10(root='datasets/', split='unlabeled', transform=torchvision.transforms.ToTensor(), download=True)

cifar10 = DataLoader(
    "cifar10",
    cifar10_dataset,
    [0.8, 0.2, 0.0],
)
stl10 = DataLoader(
    "stl10",
    stl10_dataset,
    [0.8, 0.2, 0.0]
)

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.Fliplr(0.4),
    iaa.Flipud(0.3),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
        mode=ia.ALL
    ),
    iaa.Sometimes(0.05, iaa.Grayscale()),
], random_order=True)


def augment(original_dataset, save_path, factor=2):
    if os.path.exists(save_path):
        print("Dataset already augmented, loading...")
        augmented_dataset = torch.load(save_path)
    else:
        augmented_data = []
        for img, label in original_dataset:
            augmented_data.append((img, label))
            img_np = img.numpy().transpose((1, 2, 0))
            img_np = (img_np * 255).astype('uint8')
            imgs_np = np.array([img_np] * factor)
            imgs_aug = seq.augment_images(imgs_np).transpose((0, 3, 1, 2))
            imgs_aug = torch.tensor(imgs_aug / 255)
            for img_aug in imgs_aug:
                augmented_data.append((img_aug, label))
        
        imgs, labels = zip(*augmented_data)
        augmented_dataset = torch.utils.data.TensorDataset(torch.stack(imgs), torch.tensor(labels))
        torch.save(augmented_dataset, save_path)
    
    return augmented_dataset

cifar10_dataset_augmented = augment(cifar10_dataset, 'datasets/cifar10_augmented.pt')

cifar10_augmented = DataLoader(
    "cifar10_augmented",
    cifar10_dataset_augmented,
    [0.8, 0.2, 0.0],
)