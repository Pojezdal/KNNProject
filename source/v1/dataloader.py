import torch
import torchvision
import os
from enum import Enum
import random

class DataLoader:
    def __init__(self, dataset, ratio = [0.75, 0.2, 0.05], batch_size = 64):
        self.dataset = dataset
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * ratio) for ratio in ratio])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        print(f"Train size: {len(train_data)}")

cifar10_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
stl10_dataset = torchvision.datasets.STL10(root='datasets/', split='unlabeled', transform=torchvision.transforms.ToTensor(), download=True)

cifar10 = DataLoader(
    cifar10_dataset,
    [0.8, 0.2, 0.0],
)
stl10 = DataLoader(
    stl10_dataset,
    [0.8, 0.2, 0.0]
)

class Augmentation(Enum):
    FLIP_HORIZONTAL = 1
    FLIP_VERTICAL = 2
    ROTATE_90 = 3
    ROTATE_180 = 4
    ROTATE_270 = 5
    COLOR_JITTER = 6
    GAUSSIAN_BLUR = 7
    GRAYSCALE = 8
    SKEW = 9


def augment(original_dataset, save_path, factor=2, 
            augmentations=[Augmentation.FLIP_HORIZONTAL, Augmentation.FLIP_VERTICAL, Augmentation.ROTATE_90, Augmentation.ROTATE_180, 
                           Augmentation.ROTATE_270, Augmentation.COLOR_JITTER, Augmentation.GAUSSIAN_BLUR, Augmentation.GRAYSCALE, Augmentation.SKEW]):
    if os.path.exists(save_path):
        print("Dataset already augmented, loading...")
        augmented_dataset = torch.load(save_path)
    else:
        augmented_data = []
        for img, label in original_dataset:
            augmented_data.append((img, label))
            
            rand_augments = random.sample(augmentations, factor - 1)
            for augmentation in rand_augments:
                if augmentation == Augmentation.FLIP_HORIZONTAL:
                    augmented_data.append((torch.flip(img, [2]), label))  # Horizontal flip
                elif augmentation == Augmentation.FLIP_VERTICAL:
                    augmented_data.append((torch.flip(img, [1]), label))
                elif augmentation == Augmentation.ROTATE_90:
                    augmented_data.append((torch.rot90(img, 1, [1, 2]), label))
                elif augmentation == Augmentation.ROTATE_180:
                    augmented_data.append((torch.rot90(img, 2, [1, 2]), label))
                elif augmentation == Augmentation.ROTATE_270:
                    augmented_data.append((torch.rot90(img, 3, [1, 2]), label))
                elif augmentation == Augmentation.COLOR_JITTER:
                    augmented_data.append((torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)(img), label))
                elif augmentation == Augmentation.GAUSSIAN_BLUR:
                    augmented_data.append((torchvision.transforms.GaussianBlur(kernel_size=5)(img), label))
                elif augmentation == Augmentation.GRAYSCALE:
                    augmented_data.append((torchvision.transforms.Grayscale(num_output_channels=3)(img), label))
                elif augmentation == Augmentation.SKEW:
                    augmented_data.append((torchvision.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15)(img), label))

        
        imgs, labels = zip(*augmented_data)
        augmented_dataset = torch.utils.data.TensorDataset(torch.stack(imgs), torch.tensor(labels))
        torch.save(augmented_dataset, save_path)
    
    return augmented_dataset

cifar10_dataset_augmented = augment(cifar10_dataset, 'datasets/cifar10_augmented.pt')

cifar10_augmented = DataLoader(
    cifar10_dataset_augmented,
    [0.8, 0.2, 0.0],
)