import numpy as np
from torchvision import transforms
from compressai.zoo import bmshj2018_hyperprior
import torch
import os
import time
from metrics import *
import pandas as pd
import math

model_paths = #... (list of model paths)
quality_levels = #... (list of quality levels)
npz_test_path = #... (path to npz files)

for model_path, quality_level in zip(model_paths, quality_levels):

    recons_path = #... (path to reconstruction depending on quality level)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = bmshj2018_hyperprior(quality=5, pretrained=False)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    net.to(device)

    ## Functions
    def compute_bpp(output_net):
        size = output_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in output_net['likelihoods'].values()).item()

    df = pd.DataFrame(columns=["input_filename", "model", "quality_level", "bpp_original", "q_bpp", "mean_eucl_dist","psnr", "total_time", "compression_time", "decompression_time", "input_shape"])

    for index, npz_file in enumerate(os.listdir(npz_test_path)):
        if npz_file.endswith(".npz"):
            print(f"{index} / {len(os.listdir(npz_test_path))}", end="\r")
            npz_path = os.path.join(npz_test_path, npz_file)
            
            file = open(npz_path, 'rb')
            data = np.load(file)
            cloud_2d = data['kitti_cloud_2d'][:,:,0:3]
            cloud_2d_resized = np.resize(cloud_2d, (64,512,3))

            input_tensor = transforms.ToTensor()(cloud_2d_resized).unsqueeze(0).to(device, dtype=torch.float)
            image_memory_size = os.stat(npz_path).st_size / 1024
            bpp_original = round(image_memory_size / (512 * 64), 4)

            with torch.no_grad():
                output_net = net.forward(input_tensor)

                start_complete_time = time.time()
                start_compress_time = time.time()
                compress = net.compress(input_tensor)
                stop_compress_time = time.time()
                start_decompress_time = time.time()
                decompress = net.decompress(compress["strings"], compress["shape"])
                stop_decompress_time = time.time()
                stop_complete_time = time.time()

            output_tensor = output_net['x_hat']

            output_cloud_2d = np.asarray(output_tensor.squeeze().cpu())
            recon_swapped = np.swapaxes(np.swapaxes(output_cloud_2d, 0, 2), 0,1)
            diff = np.asarray(torch.mean((output_tensor - input_tensor).abs(), axis=1).squeeze().cpu())

            np.savez(os.path.join(recons_path, npz_file), recon_swapped)

            compress_time = round(stop_compress_time - start_compress_time, 4)
            decompress_time = round(stop_decompress_time - start_decompress_time, 4)
            total_time = round(stop_complete_time - start_complete_time, 4)


            psnr = round(compute_psnr(input_tensor, output_tensor), 4)
            mean_eucl_dist = round(diff.mean(), 4)        
            bpp = round(compute_bpp(output_net), 4)

            data_dict = {"input_filename": npz_file, 
                            "model": "VAE", 
                            "quality_level": quality_level,
                            "input_shape": "(512, 64)", 
                            "bpp_original": bpp_original, 
                            "q_bpp": bpp, 
                            "mean_eucl_dist": mean_eucl_dist,
                            "psnr": psnr, 
                            "total_time": total_time, 
                            "compression_time": compress_time, 
                            "decompression_time": decompress_time}

            df = df.append(data_dict, ignore_index=True)

    df.to_csv(f"./evaluation_results_VAE_{quality_level}_pointclouds.csv")




