import torch
import torchvision

class DataLoader:
    def __init__(self, dataset, ratio = [0.75, 0.2, 0.05], batch_size = 64):
        self.dataset = dataset
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset) * ratio) for ratio in ratio])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


cifar10 = DataLoader(
    torchvision.datasets.CIFAR10(root='datasets/', train=True, download=True, transform=torchvision.transforms.ToTensor()),
)