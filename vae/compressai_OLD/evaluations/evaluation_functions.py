import torch
from PIL import Image
import math
import io
import numpy as np
from pytorch_msssim import ms_ssim

def pillow_encode(input_image, fmt="jpeg", quality=10):
    """Encode image with PIL using specified method.

    Args:
        input_image (PIL.Image): image to encode
        fmt (str, optional): formate to use for encoding. Defaults to "jpeg".
        quality (int, optional): encoding quality. Defaults to 10.

    Returns:
        PIL.image, float: encoded image, bits per pixel value
    """
    tmp = io.BytesIO()
    input_image.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (input_image.size[0] * input_image.size[1])
    rec = Image.open(tmp)
    return rec, bpp

def compute_bpp(output_net):
    """Compute bpp of VAE reconstruction.

    Args:
        output_net (torch.model): model that has been used for reconstruction.

    Returns:
        float: bits per pixel value
    """
    size = output_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in output_net['likelihoods'].values()).item()

def find_closest_bpp(target, input_image, fmt="jpeg"):
    """Find closest closest encoding for a given bpp with specified method.

    Args:
        target (float): target bpp
        input_image (PIL.image): image to be encoded
        fmt (str, optional): format to use for encoding. Defaults to "jpeg".

    Returns:
        PIL.image, float: encoded image, bits per pixel value
    """
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(input_image, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp

def find_closest_psnr(target, input_image, fmt="jpeg"):
    """Find closest closest encoding for a given PSNR value with specified method.

    Args:
        target (float): target PSNR value
        input_image (PIL.image): image to be encoded
        fmt (str, optional): format to use for encoding. Defaults to "jpeg".

    Returns:
        PIL.image, float, float: encoded image, bits per pixel value, PSNR value
    """
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _psnr(a, b):
        a = np.asarray(a).astype(np.float32)
        b = np.asarray(b).astype(np.float32)
        mse = np.mean(np.square(a - b))
        return 20*math.log10(255.) -10. * math.log10(mse)
    
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(input_image, fmt=fmt, quality=int(mid))
        psnr_val = _psnr(rec, input_image)
        if psnr_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, psnr_val

def find_closest_msssim(target, input_image, fmt="jpeg"):
    """Find closest closest encoding for a given MS-SSIM with specified method.

    Args:
        target (float): target MS-SSIM
        input_image (PIL.image): image to be encoded
        fmt (str, optional): format to use for encoding. Defaults to "jpeg".

    Returns:
        PIL.image, float: encoded image, bits per pixel value, MS-SSIM value
    """
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _mssim(a, b):
        a = torch.from_numpy(np.asarray(a).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(np.asarray(b).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return ms_ssim(a, b, data_range=255.).item()

    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(input_image, fmt=fmt, quality=int(mid))
        msssim_val = _mssim(rec, input_image)
        if msssim_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, msssim_val

def compute_psnr(a, b, max_val=255.):
    """Compute PSNR value for given inputs.

    Args:
        a (np.array): image 1
        b (np.array): image 2
        max_val (float, optional): maximum value of image. Defaults to 255..

    Returns:
        float: PSNR value
    """
    mse = torch.mean((a - b)**2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compute_msssim(a, b):
    """Compute MS-SSIM value for given inputs.

    Args:
        a (np.array): image 1
        b (np.array): image 2

    Returns:
        float: MS-SSIM value
    """
    ms_ssim_score = ms_ssim(a, b, data_range=1.).item()
    return ms_ssim_score

def compute_lpips(a, b, net="alex"):
    """Compute LPIPS value for given inputs using specified network.

    Args:
        a (np.array): image 1
        b (np.array): image 2
        net (str, optional): Name of network to use. Defaults to "alex".

    Returns:
        float: LPIPS value
    """
    loss_fn = lpips.LPIPS(net=net)
    lpips_score = loss_fn.forward(a.cpu(), b.cpu())
    return lpips_score.item()