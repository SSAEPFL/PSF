import os
import numpy as np
from astropy.io import fits

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
    # flat = flat + np.abs(np.min(flat)) * np.ones(shape=np.shape(flat)) # to avoid a zero, no difference in final result
    flat = flat / np.mean(flat)
    clean_image = (img - dark) / flat # source: http://spiff.rit.edu/classes/phys445/lectures/darkflat/darkflat.html
    return clean_image

def averageImages(path_to_images):
    """ Average the images
    : path_to_images: list of strings indicating the paths to .fit images

    return: matrix of image
    """
    N = len(path_to_images)
    with fits.open(path_to_images[0]) as file:
        final_img = np.ndarray(shape=np.shape(file[0].data))
    for path in path_to_images:
        with fits.open(path) as file:
            final_img += file[0].data/N
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

def signalToNoiseRatio(img):
    """ Compute the signal to noise ratio of an image
        Source: http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/instruments/L14/index.html
    : img: matrix of img (calibrated is best)
    return: scalar between 0 and 1
    """
    return 0
