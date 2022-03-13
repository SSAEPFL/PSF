import numpy as np
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from astropy.convolution import Moffat2DKernel
from scipy.signal import convolve as scipy_convolve
plt.style.use(astropy_mpl_style)

def calibrate(img, bias, dark, flat):
    #clean_image = (img - dark) * np.mean(flat - dark) / (flat - dark) # source: https://en.wikipedia.org/wiki/Flat-field_correction
    #clean_image = (img - flat - bias) / np.mean(dark)
    flat = flat / np.mean(flat)
    clean_image = (img - dark) / flat # source: http://spiff.rit.edu/classes/phys445/lectures/darkflat/darkflat.html
    return clean_image