"""Evaluation of VAE compression for KITTI dataset. Saves results in csv files. 

    Use hardcoded parameters since argparse is not useful here.
"""

import torch
from torchvision import transforms
from PIL import Image
import lpips
from compressai.zoo import bmshj2018_hyperprior
from utils.metrics import *
import time
import pandas as pd
import os
from utils.evaluation_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Prarameters
net_paths = #.. (list of model paths to evaluate)
quality_levels = #.. (list of qulity level strings corresponding to models)
image_folder_path = #.. (path to original image folder)
input_shapes = [(256,256)]
save_images = True
save_images_folder = #.. (path for reconstructions if you want to save them)
loss_fn_lpips = lpips.LPIPS(net="alex")


if __name__ == "__main__":
    for quality_level, net_path in zip(quality_levels, net_paths):
        ## Load Net
        net = bmshj2018_hyperprior(quality=5, pretrained=False)
        net.load_state_dict(torch.load(net_path))
        net.eval()
        net.to(device)
        print("######################")
        print(f"Quality: {quality_level} | Net: {net_path}")

        for input_shape in input_shapes:
            df = pd.DataFrame(columns=["input_filename", "model", "quality_level", "bpp_original", "q_bpp", "lpips","psnr", "ms_ssim", "mse","total_time", "compression_time", "decompression_time", "input_shape"])
            print(f"Input Shape: {input_shape}")
            count = 0
            total_images = len([x for x in os.listdir(image_folder_path)])

            for image_filename in os.listdir(image_folder_path):
                if image_filename.endswith(".png"):
                    count = count + 1
                    image_path = os.path.join(image_folder_path, image_filename)
                    ## Import Image
                    image_memory_size = os.stat(image_path).st_size / 1024
                    original_image = Image.open(image_path).convert('RGB') 
                    shape_original = original_image.size 
                    input_image = original_image.resize(input_shape)  
                    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
                    bpp_original = round(image_memory_size / (shape_original[0] * shape_original[1]), 4)

                    ## Inference
                    with torch.no_grad():
                        output_net = net.forward(input_tensor)
                    output_net['x_hat'].clamp_(0, 1)
                    output_tensor = output_net['x_hat']
                    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())
                    if save_images:
                        save_folder = os.path.join(save_images_folder, quality_level)
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        output_image.save(os.path.join(save_images_folder, quality_level, image_filename))

                    psnr = round(compute_psnr(input_tensor, output_tensor), 4)
                    msssim = round(compute_msssim(input_tensor, output_tensor), 4)
                    bpp = round(compute_bpp(output_net), 4)
                    lpips = round(compute_lpips(input_tensor, output_tensor, loss_fn_lpips), 6)
                    mse = round(compute_mse(input_tensor, output_tensor), 6)

                    print(f"{count} / {total_images} || MSE: {mse} | MS-SSIM: {msssim} | BPP: {bpp} | psnr: {psnr} | LPIPS: {lpips} |", end="\r")

                    ## Measure Time
                    with torch.no_grad():
                        start_complete_time = time.time()
                        start_compress_time = time.time()
                        compress = net.compress(input_tensor)
                        stop_compress_time = time.time()
                        start_decompress_time = time.time()
                        decompress = net.decompress(compress["strings"], compress["shape"])
                        stop_decompress_time = time.time()
                        stop_complete_time = time.time()

                    compress_time = round(stop_compress_time - start_compress_time, 4)
                    decompress_time = round(stop_decompress_time - start_decompress_time, 4)
                    total_time = round(stop_complete_time - start_complete_time, 4)
                    data_dict = {"input_filename": image_filename, 
                                    "model": "VAE", 
                                    "quality_level": quality_level,
                                    "input_shape": input_shape, 
                                    "bpp_original": bpp_original, 
                                    "q_bpp": bpp, 
                                    "lpips": lpips,
                                    "psnr": psnr, 
                                    "ms_ssim": msssim, 
                                    "mse": mse,
                                    "total_time": total_time, 
                                    "compression_time": compress_time, 
                                    "decompression_time": decompress_time}

                df = df.append(data_dict, ignore_index=True)

            df.to_csv(f"./evaluation_results_VAE_{quality_level}_{input_shape}.csv")