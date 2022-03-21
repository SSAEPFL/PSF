import os
import numpy as np
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from astropy.convolution import Moffat2DKernel
from scipy.signal import convolve as scipy_convolve
plt.style.use(astropy_mpl_style)

def calibrate(img, bias, dark, flat):
    """ Image (img) calibration using a bias frame, a dark frame and a flat frame
    : img: raw image of interest
      bias: bias image (shortest exposure of nothing) --> readout noise
      dark: dark image (same exposure time as img, but of nothing) --> thermal noise
      flat: flat image (image to correct for vignetting)
      
      return: matrix of image
    """
    #clean_image = (img - dark) * np.mean(flat - dark) / (flat - dark) # source: https://en.wikipedia.org/wiki/Flat-field_correction
    #clean_image = (img - flat - bias) / np.mean(dark)
    flat = flat - bias
    flat = flat + np.ones(shape=np.shape(flat1)) # to avoid a zero, no difference in final result
    flat = flat / np.mean(flat)
    clean_image = (img - dark) / flat # source: http://spiff.rit.edu/classes/phys445/lectures/darkflat/darkflat.html
    return clean_image

def averageImages(path_to_images):
    """ Average the images
    : path_to_images: list of strings indicating the paths to .fit images

    return: matrix of image
    """
    N = len(path_to_images)
    with fits.getdata(path[0]) as img:
        final_img = np.ndarray(shape=np.shape(img))
    for path in path_to_images:
        with fits.getdata(path) as img:
            final_img += img/N
    return final_img

def averageFolder(path_to_folder):
    """ Average image of all the .fit images in folder
    : path_to_folder: string of path to folder

    return: matrix of image
    """
    path_to_images = []
    for file in os.listdir(path_to_folder):
        if file.endswith(".fit"):
            path_to_images.append(os.path.join(path_to_folder, file))
    return averageImages(path_to_images)
