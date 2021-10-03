import torch
from PIL import Image
import math
import io
import numpy as np
from pytorch_msssim import ms_ssim

def pillow_encode(input_image, fmt="jpeg", quality=10):
    tmp = io.BytesIO()
    input_image.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (input_image.size[0] * input_image.size[1])
    rec = Image.open(tmp)
    return rec, bpp

def compute_bpp(output_net):
    size = output_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in output_net['likelihoods'].values()).item()

def find_closest_bpp(target, input_image, fmt="jpeg"):
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

def find_closest_bpp(target, input_image, fmt="jpeg"):
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
    mse = torch.mean((a - b)**2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr

def compute_msssim(a, b):
    ms_ssim_score = ms_ssim(a, b, data_range=1.).item()
    return ms_ssim_score

def compute_lpips(a, b, net="alex"):
    loss_fn = lpips.LPIPS(net=net)
    lpips_score = loss_fn.forward(a.cpu(), b.cpu())
    return lpips_score.item()