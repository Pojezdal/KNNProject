import time
import torch
import torch.nn as nn
import torch.optim as optim
import piq
from dataloader import cifar10
from train_stats import TrainStats
from scheduler_wrapper import SchedulerWrapper

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

## This class is the encoder part of the VAE. It takes an input image and compresses it into a latent space representation.
## The latent space is represented by the maeans and logvariances of the normal distribution that the latent space is assumed to follow.
class Encoder(nn.Module):
    def __init__(self, shared_layers, mean_layers, logvar_layers):
        super(Encoder, self).__init__()
        self.shared_layers = shared_layers
        self.mean_layers = mean_layers
        self.logvar_layers = logvar_layers
        
    def default(compressed_img_size, input_channels, latent_size, hidden_sizes = None) -> 'Encoder':
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128, 256]
        
        shared_layers = nn.Sequential()
        for hidden_size in hidden_sizes:
            shared_layers.append(nn.Conv2d(input_channels, hidden_size, kernel_size=3, stride=2, padding=1))
            shared_layers.append(nn.BatchNorm2d(hidden_size))
            shared_layers.append(nn.LeakyReLU())
            input_channels = hidden_size
        mean_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(compressed_img_size * compressed_img_size * hidden_sizes[-1], latent_size),
            #nn.LeakyReLU(),
        )
        logvar_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(compressed_img_size * compressed_img_size * hidden_sizes[-1], latent_size),
            #nn.LeakyReLU(),
        )
        return Encoder(shared_layers, mean_layers, logvar_layers)
        
        
    def forward(self, x):
        x = self.shared_layers(x)
        mean = self.mean_layers(x)
        logvar = self.logvar_layers(x)
        return mean, logvar
    

## This class is the decoder part of the VAE. It takes a latent space representation and reconstructs the original image.
class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = layers.to(device)
        
    def default(compressed_img_size, latent_size, output_channels, hidden_sizes = None) -> 'Decoder':
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64, 32]
        else:
            hidden_sizes = hidden_sizes[::-1]
        
        layers = nn.Sequential(
            nn.Linear(latent_size, compressed_img_size * compressed_img_size * hidden_sizes[0]),
            nn.Unflatten(1, (hidden_sizes[0], compressed_img_size, compressed_img_size))
        )
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.ConvTranspose2d(hidden_sizes[i], hidden_sizes[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.BatchNorm2d(hidden_sizes[i + 1]))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.ConvTranspose2d(hidden_sizes[-1], hidden_sizes[-1], kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.BatchNorm2d(hidden_sizes[-1]))
        layers.append(nn.LeakyReLU())
        
        layers.append(nn.Conv2d(hidden_sizes[-1], output_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        
        return Decoder(layers)
        
        
    def forward(self, x):
        x = self.layers(x)
        return x

## This class is the VAE itself. It is composed of an encoder and a decoder.
class VAE(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder):
        super(VAE, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        
        
    def default(compressed_img_size, input_channels : int, latent_size : int, hidden_sizes = None) -> 'VAE':
        encoder = Encoder.default(compressed_img_size, input_channels, latent_size, hidden_sizes).to(device)
        decoder = Decoder.default(compressed_img_size, latent_size, input_channels, hidden_sizes).to(device)
        return VAE(encoder, decoder)
        
        
    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x):
        #print(x.shape)
        mean, logvar = self.encoder(x)
        z = self.reparametrize(mean, logvar)
        #print(z.shape)
        x_recon = self.decoder(z)
        #print(x_recon.shape)
        return x_recon, mean, logvar


## This function is the loss function of the VAE. It is composed of a comparison loss and a Kullback-Leibler divergence loss.
def loss_function(comp_loss_fn, recon_x, x, mu, logvar, kld_weight=0.00025):
    comp_loss = comp_loss_fn(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return comp_loss + kld_weight * kld_loss


## This function is used to train the VAE. It takes the model, the number of epochs, the data loader, the comparison loss function, 
## the optimizer, the scheduler, the Kullback-Leibler divergence weight and the image interval as arguments.
def train(model, num_epochs, data_loader, comp_loss_fn, optimizer, scheduler=None, kld_weight=0.00025, img_interval=1):
    stats = TrainStats(num_epochs)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        for inputs, _ in data_loader.train_loader:
            inputs = inputs.to(device)            
            
            outputs, mean, logvar = model(inputs)
            
            loss = loss_function(comp_loss_fn, outputs, inputs, mean, logvar, kld_weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        total_error = 0.0
        data_size = 0
        with torch.no_grad():
            for input, _ in data_loader.val_loader:
                input = input.to(device)
                output, _, _ = model.forward(input)
                loss = loss_function(comp_loss_fn, output, input, mean, logvar, kld_weight)
                total_error += loss.item()
                data_size += input.size(0)
            
        if scheduler is not None:
            scheduler.step(loss)

            mean_loss = total_error / data_size
            stats.losses.append((epoch, mean_loss))
            if (epoch + 1) % img_interval == 0:
                stats.add_image(output)
            epich_time = time.time() - start_time
            stats.total_time += epich_time
            
            print(f"Epoch {epoch+1}/{num_epochs}, Validation error: {mean_loss:.4f}, Time: {epich_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.8f}")
    
    stats.add_image(input)
    
    return stats


model = VAE.default(2, 3, 128)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler1 = SchedulerWrapper(optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.94))
scheduler2 = SchedulerWrapper(optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, threshold=0.01))
loss_fn = nn.MSELoss(reduction='sum')
loss_fn2 = piq.SSIMLoss(kernel_size=5, reduction='sum')

stats = train(model, 10, cifar10, loss_fn, optimizer, scheduler1, img_interval=5)

print(stats)
stats.plot_loss("files/loss.png")
stats.show_images("files/reconstruction.png")

