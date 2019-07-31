from io import BytesIO
import PIL.Image
import IPython.display
import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
from scipy.ndimage.filters import gaussian_filter, median_filter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
    
def autotone_PIL(img):
    img = np.array(img)
    img[:,:,0] = np.interp(img[:,:,0], [np.amin(img[:,:,0]), np.amax(img[:,:,0])], [0, 255])
    img[:,:,1] = np.interp(img[:,:,1], [np.amin(img[:,:,1]), np.amax(img[:,:,1])], [0, 255])
    img[:,:,2] = np.interp(img[:,:,2], [np.amin(img[:,:,2]), np.amax(img[:,:,2])], [0, 255])
    img = PIL.Image.fromarray(img)
    return img

def deprocess(tensor):
    # Remove batch dimension
    # Shape before: BCSS
    tensor = tensor.data.squeeze() 
        
    # To CPU and numpy array
    img = tensor.cpu().numpy() 
    
    # Channels last
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    # Clip to visible range
    img = np.clip(img, 0., 1.) # Clip to 0./1. range
    
    # 0./1. range to 0./255. range
    img *= 255. 
        
    # To PIL image format
    img = PIL.Image.fromarray(img.astype(np.uint8))  
    
    return img

def filter_median(npimg, params):
    npimg = median_filter(npimg, size=(1, 1, params['fsize'], params['fsize']))  
    return npimg
    
def filter_TV(npimg, params):
    npimg = pytorch_to_skimage(npimg)
    npimg = denoise_tv_chambolle(npimg, weight=0.1, multichannel=True)
    npimg = skimage_to_pytorch(npimg)
    return npimg

def gray_square_PIL(size):
    # Gray square, -1./1. range
    img = np.random.normal(0, 0.01, (size, size, 3)) 
    
    # -1./1. range to 0./255. range
    img /= 2.
    img += 0.5
    img *= 255.

    # To PIL image format
    img = PIL.Image.fromarray(img.astype(np.uint8))
    
    return img

def show_img(img, fmt='jpeg'):
    if type(img) is np.ndarray:
        img = PIL.Image.fromarray(img)
    f = BytesIO()
    img.save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
def pytorch_to_skimage(img):
    # No batch dimension
    img = img[0]
    # Channels last
    img = np.swapaxes(img, 0, 2)
    return img
    
def skimage_to_pytorch(img):
    # Channels first
    img = np.swapaxes(img, 0, 2)
    # Skimage uses double
    img = img.astype(np.float32)
    # No Batch dimension
    img = np.expand_dims(img, 0)
    return img

def np_PIL(img):
    return PIL.Image.fromarray(img)

def PIL_np(img):
    return np.array(img)