import compressai
import torch
from dataloader import cifar10, cifar10_augmented, stl10
from train_stats import TrainStats
from torchinfo import summary

def eval(model, data_loader, loss_fn):
    stats = TrainStats(0)

    model.eval()
    total_error = 0.0
    data_size = 0
    total_bitrate = 0.0
    with torch.no_grad():
        for input, _ in data_loader.val_loader:
            output = model.forward(input)
            likelihoods = output['likelihoods']
            output = output['x_hat']
            loss = loss_fn(output, input)
            total_error += loss.item()
            data_size += input.size(0)
            total_likelihoods = likelihoods['y']
            bitrate = torch.sum(total_likelihoods).item() / (torch.log(torch.tensor(2.)) * input.numel())
            total_bitrate += bitrate
        
    mean_loss = total_error / data_size
    mean_bitrate = total_bitrate / data_size
    print(f"Mean loss: {mean_loss:.4f}")
    print(f"Mean bitrate: {mean_bitrate:.4f}")
    
    stats.add_image(output)
    stats.add_image(input)
    
    return stats


model = compressai.zoo.cheng2020_attn(6, pretrained=True)

summary(model, (1, 3, 32, 32))

#stats = eval(model, stl10, torch.nn.MSELoss(reduction='sum'))

#stats.show_images()