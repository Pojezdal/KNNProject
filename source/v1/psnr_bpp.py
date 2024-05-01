import compressai
import torchvision.transforms as transforms 
import piq
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import io
from dataloader import DataLoader
from dataloader import stl10_dataset

t = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256)),
])

stl10 = DataLoader(
    "stl10",
    stl10_dataset,
    [0.8, 0.2, 0.0],
    batch_size=1
)

def psnr_bpp(model_type, image, start_level=1, end_level=6):
    bpps = []
    psnrs = []
    tensor_image = image[0]
    image_size = tensor_image.shape[2] * tensor_image.shape[3]
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
    tensor_image = image[0][0]
    image_size = tensor_image.shape[1] * tensor_image.shape[2]
    for quality in range(start_level, end_level + 1):
        with io.BytesIO() as f:
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(tensor_image)
            pil_img.save(f, format='JPEG', quality=quality * 10)
            f.seek(0)
            jpeg_image = Image.open(f)
            
            compressed_size = len(f.getvalue()) * 8
            bpp = compressed_size / image_size
            bpps.append(bpp)

            # Calculate the PSNR.
            decompressed_image = t(jpeg_image).unsqueeze(0)
            psnr = piq.psnr(tensor_image.unsqueeze(0), decompressed_image).item()
            psnrs.append(psnr)
            
            #jpeg_image.close()

    return bpps, psnrs
        

def plot_psnr_bpp(model_metrics):
    # plot bpp vs psnr
    plt.xlabel('BPP (bits per pixel)')
    plt.ylabel('PSNR')
    plt.title('BPP vs PSNR')
    for model, metrics in model_metrics.items():
        plt.plot(metrics["bpp"], metrics["psnr"], 'o-', markersize=4, label=model)
    plt.legend()
    plt.show()


#image_names = [img_name for img_name in glob(image_folder + '/*.png')] # ['datasets/kodak/kodim01.png']
batch_count = 500 #==images if batch_size = 1
metrics = { "bms": {'bpp': [0]*8, 'psnr':[0]*8}, "jpeg": {'bpp': [0]*9, 'psnr':[0]*9} }
for i, image in enumerate(stl10.val_loader):
    print("Processing image ", i)
    for level, (bpp, psnr) in enumerate(zip(*psnr_bpp(compressai.zoo.bmshj2018_factorized, image, 1, 8))):
        # if 'bpp' not in metrics["bms"][level]:
        #     metrics["bms"][level]['bpp'] = 0
        # if 'psnr' not in metrics["bms"][level]:
            # metrics["bms"][level]['psnr'] = 0
        metrics["bms"]['bpp'][level] += bpp
        metrics["bms"]['psnr'][level] += psnr
    # for bpp, psnr in zip(*psnr_bpp(compressai.zoo.cheng2020_attn, image, 1, 6)):
    #     if bpp not in metrics["cheng"]:
    #         metrics["cheng"][bpp] = 0
    #     metrics["cheng"][bpp] += psnr
    for level, (bpp, psnr) in enumerate(zip(*psnr_bpp_jpeg(image, 1, 9))):
        # if 'bpp' not in metrics["jpeg"][level]:
        #     metrics["jpeg"][level]['bpp'] = 0
        # if 'psnr' not in metrics["jpeg"][level]:
        #     metrics["jpeg"][level]['psnr'] = 0
        metrics["jpeg"]['bpp'][level] += bpp
        metrics["jpeg"]['psnr'][level] += psnr
    if i == batch_count:
        break
    #image.close()

for _, model_stats in metrics.items():
    for metric, values in model_stats.items():
        for level in range(len(values)):
            model_stats[metric][level] /= batch_count

# log metrics into json
import json
with open('metrics_new.json', 'w') as f:
    json.dump(metrics, f)


plot_psnr_bpp(metrics)