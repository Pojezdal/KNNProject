import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

class TrainStats:
    def __init__(self, epochs):
        self.total_time = 0.0
        self.total_epochs = epochs
        self.losses = []
        self.images = torch.Tensor()
    
    def add_image(self, images):
        self.images = torch.cat((self.images, images), dim=2)
    
    def show_images(self, filename = None):
        npimage = torchvision.utils.make_grid(self.images).cpu().numpy()
        npimage = npimage.transpose(1, 2, 0)
        plt.imshow(npimage)
        if filename:
            plt.imsave(filename, npimage)
        plt.show()
        
    def plot_loss(self, filename = None):
        plt.plot(*zip(*self.losses))
        plt.xticks(np.linspace(0, self.total_epochs - 1, num=10))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        if filename:
            plt.savefig(filename)
        plt.show()
        
    def __str__(self):
        total_time_formatted = (f"{int(self.total_time):d} seconds" if self.total_time < 60 else 
                                f"{int(self.total_time // 60):d}:{int(self.total_time % 60):02d} minutes" if self.total_time < 3600 else
                                f"{int(self.total_time // 3600):d}:{int((self.total_time % 3600) // 60):02d}:{int(self.total_time % 60):02d} hours")
        s =  f"*** Statistics ***\n"
        s += f"Total time: {total_time_formatted}\nTime per epoch: {(self.total_time / self.total_epochs):.2f} seconds\n"
        s += f"Final loss: {self.losses[-1][1]:.5f} after {self.total_epochs} epochs\n"
        s += f"******************\n"
        return s
