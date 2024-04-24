import compressai
import torchvision.transforms as transforms 
import piq
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import io

t = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256)),
])

def psnr_bpp(model_type, image, start_level=1, end_level=6):
    bpps = []
    psnrs = []
    image_size = image.size[0] * image.size[1]
    tensor_image = t(image).unsqueeze(0)
    for level in range(start_level, end_level + 1):
        model = model_type(level, pretrained=True)
        compressed_image = model.compress(tensor_image)
        model.forward(tensor_image)

        compressed_size = 0
        for string in compressed_image["strings"]:
            compressed_size += len(string[0]) * 8

        bpps.append(compressed_size / image_size)

        decompressed_image = model.decompress(compressed_image["strings"], compressed_image["shape"])["x_hat"]
        psnr = piq.psnr(tensor_image, decompressed_image).item()
        psnrs.append(psnr)

    return bpps, psnrs

def psnr_bpp_jpeg(image, start_level=1, end_level=9):
    bpps = []
    psnrs = []
    image_size = image.size[0] * image.size[1]
    tensor_image = t(image).unsqueeze(0)
    for quality in range(start_level, end_level + 1):
        with io.BytesIO() as f:
            image.save(f, format='JPEG', quality=quality * 10)
            f.seek(0)
            jpeg_image = Image.open(f)
            
            compressed_size = len(f.getvalue()) * 8
            bpp = compressed_size / image_size
            bpps.append(bpp)

            # Calculate the PSNR.
            decompressed_image = t(jpeg_image).unsqueeze(0)
            psnr = piq.psnr(tensor_image, decompressed_image).item()
            psnrs.append(psnr)
            
            jpeg_image.close()

    return bpps, psnrs
        

def plot_psnr_bpp(stats):
    plt.xlabel('BPP (bits per pixel)')
    plt.ylabel('PSNR')
    plt.title('BPP vs PSNR')
    for model_name, model_stats in stats.items():
        bpps = model_stats.keys()
        psnrs = model_stats.values()
        plt.plot(bpps, psnrs, 'o-', markersize=4, label=model_name)
    plt.legend()
    plt.show()


image_folder = 'datasets/kodak'
image_names = [img_name for img_name in glob(image_folder + '/*.png')] # ['datasets/kodak/kodim01.png']
metrics = { "bms": {}, "cheng": {}, "jpeg": {} }
for file_name in image_names:
    print(f"Processing {file_name}")
    image = Image.open(file_name)
    for bpp, psnr in zip(*psnr_bpp(compressai.zoo.bmshj2018_factorized, image, 1, 8)):
        if bpp not in metrics["bms"]:
            metrics["bms"][bpp] = 0
        metrics["bms"][bpp] += psnr
    for bpp, psnr in zip(*psnr_bpp(compressai.zoo.cheng2020_attn, image, 1, 6)):
        if bpp not in metrics["cheng"]:
            metrics["cheng"][bpp] = 0
        metrics["cheng"][bpp] += psnr
    for bpp, psnr in zip(*psnr_bpp_jpeg(image, 1, 9)):
        if bpp not in metrics["jpeg"]:
            metrics["jpeg"][bpp] = 0
        metrics["jpeg"][bpp] += psnr
    image.close()

for _, model_stats in metrics.items():
    for bpp, psnr in model_stats.items():
        model_stats[bpp] /= len(image_names)

plot_psnr_bpp(metrics)