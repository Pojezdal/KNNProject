import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, 'PerceptualSimilarity')

import lpips

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Define the training dataset
full_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, download=True, transform=transform)

# Define the size of the validation set (e.g., 20% of the entire dataset)
validation_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - validation_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, validation_size])

# Create DataLoader instances for training and validation
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

input_dim = 32 * 32 * 3

# Define the size of the encoding (latent space)
encoding_dim = 16 * 32 * 3

# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()  # Sigmoid activation for binary data
        )

    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x

# Define a function to display images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize if normalization was applied during transformation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Instantiate the autoencoder model
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
#criterion = lpips.LPIPS(net='alex', spatial=True)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

images = None
# Assuming you have a PyTorch DataLoader with your dataset, you can iterate over it and train the autoencoder as follows:
for epoch in range(5):
    print(f"Epoch {epoch+1}")
    for data in train_loader:
        inputs, _ = data
        #inputs = inputs.view(-1, input_dim)  # Flatten the input if needed
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        #loss = criterion(inputs.view(inputs.size(0), 3, 32, 32), outputs.view(outputs.size(0), 3, 32, 32)).sum()
        loss.backward()
        optimizer.step()
        
    autoencoder.eval()
    total_squared_error = 0.0
    nb_data = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, _ = data
            #inputs = inputs.view(-1, input_dim)
            y = autoencoder(inputs)
            total_squared_error += torch.nn.functional.mse_loss(y, inputs, reduction='sum')
            #total_squared_error += criterion(inputs.view(inputs.size(0), 3, 32, 32), y.view(y.size(0), 3, 32, 32)).sum()
            nb_data += len(inputs)
    
    print(f"MSE: {total_squared_error / nb_data}")
    
    if images is None:
        images = torch.cat((inputs.view(inputs.size(0), 3, 32, 32), y.view(y.size(0), 3, 32, 32)), dim=2)
    else:
        images = torch.cat((images, y.view(y.size(0), 3, 32, 32)), dim=2)

imshow(torchvision.utils.make_grid(images))
        
