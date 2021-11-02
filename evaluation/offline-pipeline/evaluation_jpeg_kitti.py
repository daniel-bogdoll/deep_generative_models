"""
Evaluation of JPEG compression for KITTI dataset. Saves results in csv files. 
usage: evaluation_jpeg_kitti.py [-h] --image_folder_path IMAGE_FOLDER_PATH
                                [--input_shape INPUT_SHAPE]
                                [--save_images SAVE_IMAGES]
                                [--save_images_folder SAVE_IMAGES_FOLDER]
"""

import torch
from torchvision import transforms
from PIL import Image
import lpips
from utils.metrics import *
import pandas as pd
import os
from utils.evaluation_functions import *
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argparser
parser = argparse.ArgumentParser(description='Evaluation of JPEG compression for KITTI dataset. Saves results in csv files.')
parser.add_argument('--image_folder_path', type=str, required=True,
                    help='Folder with images to encode (KITTI folder)')
parser.add_argument('--input_shape', type=tuple, default=(256,256),
                    help='Shape for images before encoding.')
parser.add_argument('--save_images', type=str, default=False,
                    help='Save encoded images.')
parser.add_argument('--save_images_folder', type=str, default="./encoded images",
                    help='Folder for encoded images if save_images flag is set.')

args = parser.parse_args()

# Parameters
image_folder_path = args.image_folder_path
input_shape = args.input_shape
save_images = args.save_images
save_images_folder = args.save_images_folder
compression_method = "JPEG"
quality_levels = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1] 

if __name__ == "__main__":
    loss_fn_lpips = lpips.LPIPS(net="alex")
    for quality_level in quality_levels:
        print("######################")
        print(f"Quality: {quality_level}")

        df = pd.DataFrame(columns=["input_filename", "model", "quality_level", "bpp_original", "q_bpp", "lpips","psnr", "ms_ssim", "mse","total_time", "compression_time", "decompression_time", "input_shape"])
        print(f"Input Shape: {input_shape}")
        count = 0
        total_images = len([x for x in os.listdir(image_folder_path)])

        for image_filename in os.listdir(image_folder_path):
            if image_filename.endswith(".png"):
                count = count + 1
                image_path = os.path.join(image_folder_path, image_filename)

                image_memory_size = os.stat(image_path).st_size / 1024
                original_image = Image.open(image_path).convert('RGB') 
                shape_original = original_image.size 
                input_image = original_image.resize(input_shape)  
                input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
                bpp_original = round(image_memory_size / (shape_original[0] * shape_original[1]), 4)

                # Calculate JPEG reconstruction
                rec_jpeg, bpp_jpeg = find_closest_bpp(quality_level, input_image)
                rec_jpeg_tensor = transforms.ToTensor()(rec_jpeg).unsqueeze(0).to(device)

                psnr = round(compute_psnr(input_tensor, rec_jpeg_tensor), 4)
                msssim = round(compute_msssim(input_tensor, rec_jpeg_tensor), 4)
                lpips = round(compute_lpips(input_tensor, rec_jpeg_tensor, loss_fn_lpips), 6)
                mse = round(compute_mse(input_tensor, rec_jpeg_tensor), 6)

                print(f"{count} / {total_images} || MSE: {mse} | MS-SSIM: {msssim} | BPP: {bpp_jpeg} | psnr: {psnr} | LPIPS: {lpips} |", end="\r")

                data_dict = {"input_filename": image_filename, 
                                "model": "JPEG", 
                                "quality_level": quality_level,
                                "input_shape": input_shape, 
                                "bpp_original": bpp_original, 
                                "q_bpp": bpp_jpeg, 
                                "lpips": lpips,
                                "psnr": psnr, 
                                "ms_ssim": msssim, 
                                "mse": mse,
                                "total_time": 0.0, 
                                "compression_time": 0.0, 
                                "decompression_time": 0.0}

            df = df.append(data_dict, ignore_index=True)

        df.to_csv(f"./evaluation_results_JPEG_{quality_level}_{input_shape}.csv")