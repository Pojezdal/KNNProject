import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import dataloader

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
        images = None
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}")
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

            print(f"{self.loss_function._get_name()}: {total_error / data_size}")

            if image_interval > 0 and epoch % image_interval == 0:
                if images is None:
                    images = torch.cat((input, output), dim=2)
                else:
                    images = torch.cat((images, output), dim=2)
        
        if images is not None:
            self.show(torchvision.utils.make_grid(images))
    
    
    def show(self, images):
        npimage = images.numpy()
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        plt.show()