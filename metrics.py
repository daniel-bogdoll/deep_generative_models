import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim

import lpips


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

net="alex" # for LPIPS 
loss_fn = lpips.LPIPS(net=net)

def compute_psnr(a, b, max_val=255.):
    """Computation of PSNR

    Args:
        a ([Tensor]): Orignal image
        b ([Tensor]): Reconstructed image
        max_val ([Float], optional): [description]. Defaults to 255..

    Returns:
        [numpy.float64]: PSNR Value
    """
    
    mse = torch.mean((a - b)**2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)

    return psnr

def compute_msssim(a, b):
    """Computation of MS_SSIM with Pytorch MS-SSIM package

    Args:
        a ([Tensor]): Orignal image
        b ([Tensor]): Reconstructed image

    Returns:
        [float]: MS_SSIM Value
    """

    ms_ssim_score = ms_ssim(a, b, data_range=1.).item()

    return ms_ssim_score

def compute_lpips(a, b):
    """Computation of LIPIS with https://github.com/richzhang/PerceptualSimilarity

    Args:
        a ([Tensor]): Orignal image
        b ([Tensor]): Reconstructed image

    Returns:
        [float]: LPIPS Value
    """
    
    lpips_score = loss_fn.forward(a.cpu(), b.cpu())
    
    return lpips_score.item()

def compute_mse(a, b):
    """Computation of MSE 

    Args:
        a ([Tensor]): Orignal image
        b ([Tensor]): Reconstructed image

    Returns:
        [float]: MSE Value
    """

    mse = torch.mean((a - b)**2).item()
    
    return mse