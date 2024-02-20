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
    #nn.MSELoss(reduction='sum'),
    lpips.DSSIM(use_gpu=False,colorspace="RGB"),
    optim.Adam,
    0.001,
    optim.lr_scheduler.MultiplicativeLR,
    lambda epoch: 0.94,
)

model2 = autoencoder.Autoencoder(
    nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ),
    nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
        nn.Sigmoid(),
    ),
    #nn.MSELoss(reduction='sum'),
    lpips.LPIPS(net='alex', spatial=True).to(autoencoder.device),
    optim.Adam,
    0.01,
    optim.lr_scheduler.MultiplicativeLR,
    lambda epoch: 0.95 if epoch<=50 else 1.0
)

model2.train_model(dataloader.cifar10, 1, 1)