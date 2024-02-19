import torch.nn as nn
import torch.optim as optim
import dataloader
import autoencoder
import sys
sys.path.insert(0, 'PerceptualSimilarity')
import lpips

model = autoencoder.Autoencoder(
    nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
        nn.Sigmoid(),
    ),
    nn.MSELoss(reduction='sum'),
    #lpips.LPIPS(net='alex').to(autoencoder.device),
    optim.Adam,
    0.001,
    optim.lr_scheduler.MultiplicativeLR,
    lambda epoch: 0.94,
)

model.train_model(dataloader.cifar10, 5, 1)