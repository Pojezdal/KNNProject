# read json file containing an array of metrics

import json
import matplotlib.pyplot as plt

with open('metrics.json') as f:
    metrics = json.load(f)

# compute average bpp and psnr for each model
model_metrics = {}
for images in metrics:
    for model, image_metrics in images.items():
        if model not in model_metrics:
            model_metrics[model] = [{'bpp': 0, 'psnr': 0, 'count': 0} for _ in range(len(image_metrics))]
        for level, (bpp, psnr) in enumerate(image_metrics.items()):
            model_metrics[model][level]['bpp'] += float(bpp)
            model_metrics[model][level]['psnr'] += psnr
            model_metrics[model][level]['count'] += 1

print(model_metrics)

# compute average bpp and psnr for each model
for model, levels in model_metrics.items():
    for level, metrics in enumerate(levels):
        model_metrics[model][level]['bpp'] /= model_metrics[model][level]['count']

# plot bpp vs psnr
plt.xlabel('BPP (bits per pixel)')
plt.ylabel('PSNR')
plt.title('BPP vs PSNR')
for model, levels in model_metrics.items():
    plt.plot([level['bpp'] for level in levels], [level['psnr'] for level in levels], 'o-', markersize=4, label=model)
plt.legend()
plt.show()



