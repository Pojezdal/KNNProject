import torch
import torch.nn as nn
import torch.optim as optim
import dataloader
import autoencoder
import train_stats
import sys
import lpips
import time
from train_stats import TrainStats

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
    #lpips.LPIPS(net='alex', spatial=True).to(autoencoder.device),
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

def generic_train_model(model, loader : dataloader.DataLoader, epochs = 5, image_interval = 1, optimizer = optim.Adam, loss_function = nn.MSELoss(reduction='sum'), scheduler = None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    stats = TrainStats(epochs)
    for epoch in range(0, epochs):
        print(f"Epoch {epoch + 1}")
        epoch_start_time = time.time()
        for input, _ in loader.train_loader:
            input = input.to(device)
            model.optimizer.zero_grad()
            output = model.forward(input)
            loss = loss_function(output, input)
            loss.backward()
            optimizer.step()

        model.eval()
        total_error = 0.0
        data_size = 0
        with torch.no_grad():
            for input, _ in loader.val_loader:
                input = input.to(device)
                output = model.forward(input)
                total_error += loss_function(output, input)
                data_size += len(input)
        if (scheduler is not None):
            scheduler.step()
        
        stats.losses.append((epoch, total_error / data_size))
        print(f"{loss_function._get_name()}: {(total_error / data_size):.5f}")
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        stats.total_time += epoch_time
        print(f"Time of the epoch: {epoch_time:.2f} seconds")

        if image_interval > 0 and epoch % image_interval == 0:
            stats.add_image(output)
    
    stats.add_image(input)        
    return stats

stats = model.train_model(dataloader.stl10, 2, 1)

print(stats)
stats.plot_loss("files/loss.png")
stats.show_images("files/images.png")