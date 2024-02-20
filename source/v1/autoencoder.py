import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import dataloader
import time
from train_stats import TrainStats

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, loss_function = nn.MSELoss(), opt_type = optim.Adam, learning_rate = 0.001, 
                 scheduler_type = None, scheduler_lambda = None):
        super().__init__()
        self.encoder = encoder_layers.to(device)
        self.decoder = decoder_layers.to(device)
        self.loss_function = loss_function
        self.optimizer = opt_type(self.parameters(), lr=learning_rate)
        self.scheduler = scheduler_type(self.optimizer, scheduler_lambda) if scheduler_type is not None else None
            
    def forward(self, input):
        #print(input.shape)
        intermediate = self.encoder(input)
        #print(intermediate.shape)
        output = self.decoder(intermediate)
        #print(output.shape)
        return output

    def train_model(self, loader : dataloader.DataLoader, epochs = 5, image_interval = 1):
        stats = TrainStats(epochs)
        for epoch in range(0, epochs):
            print(f"Epoch {epoch + 1}")
            epoch_start_time = time.time()
            for input, _ in loader.train_loader:
                input = input.to(device)
                self.optimizer.zero_grad()
                output = self(input)
                loss = self.loss_function(output, input)
                loss.backward()
                self.optimizer.step()

            self.eval()
            total_error = 0.0
            data_size = 0
            with torch.no_grad():
                for input, _ in loader.val_loader:
                    input = input.to(device)
                    output = self(input)
                    total_error += self.loss_function(output, input)
                    data_size += len(input)
            if (self.scheduler is not None):
                self.scheduler.step()
            
            stats.losses.append((epoch, total_error / data_size))
            print(f"{self.loss_function._get_name()}: {(total_error / data_size):.5f}")
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            stats.total_time += epoch_time
            print(f"Time of the epoch: {epoch_time:.2f} seconds")

            if image_interval > 0 and epoch % image_interval == 0:
                stats.add_image(output)
        
        stats.add_image(input)        
        return stats

    def compression_ratio(self, input_size):
        output_size = input_size
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                output_size = (((output_size / layer.in_channels) - layer.kernel_size[0] + 2 * layer.padding[0]) // (layer.stride[0] * layer.stride[0]) + 1) * layer.out_channels
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size - layer.kernel_size) // (layer.stride * layer.stride) + 1
            elif isinstance(layer, nn.Linear):
                output_size = layer.out_features
        return output_size / input_size