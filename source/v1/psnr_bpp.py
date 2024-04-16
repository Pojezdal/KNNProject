import compressai
import torchvision.transforms as transforms 
import piq
from PIL import Image
import matplotlib.pyplot as plt

def psnr_bpp(model_type, image, image_size, start_level=1, end_level=6):
    bpps = []
    psnrs = []

    for level in range(start_level, end_level + 1):
        model = model_type(level, pretrained=True)
        compressed_image = model.compress(image)

        compressed_size = 0
        for string in compressed_image["strings"]:
            compressed_size += len(string[0]) * 8

        bpps.append(compressed_size / image_size)

        decompressed_image = model.decompress(compressed_image["strings"], compressed_image["shape"])["x_hat"]
        psnr = piq.psnr(tensor_image, decompressed_image).item()
        psnrs.append(psnr)

    return bpps, psnrs

def plot_psnr_bpp(stats):
    plt.xlabel('BPP (bits per pixel)')
    plt.ylabel('PSNR')
    plt.title('BPP vs PSNR')
    for model_name, model_stats in stats.items():
        bpps = model_stats["bpp"]
        psnrs = model_stats["psnr"] 
        plt.plot(bpps, psnrs, 'o-', markersize=4, label=model_name)
    plt.legend()
    plt.show()


transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256)),
])

image_path = 'datasets/kodak/kodim01.png'
image = Image.open(image_path)
image_size = image.size[0] * image.size[1]
tensor_image = transforms(image).unsqueeze(0)

stats = { "bms": {}, "cheng": {} }
stats["bms"]["bpp"], stats["bms"]["psnr"] = psnr_bpp(compressai.zoo.bmshj2018_factorized, tensor_image, image_size, 1, 8)
stats["cheng"]["bpp"], stats["cheng"]["psnr"] = psnr_bpp(compressai.zoo.cheng2020_attn, tensor_image, image_size, 1, 6)

plot_psnr_bpp(stats)