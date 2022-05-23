from unittest.main import main
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os

from fonctions import *


def map_blocks_adresses(raw, column):
    """
    Maps the adresses of the individual blocks from a given raw and column identifier of the 4x8 matrix to the corresponding mosaic identifier (1 to 32) given the following convention

    |32|31|30|29|16|15|14|13|
    |28|27|26|25|12|11|10| 9|
    |24|23|22|21| 8| 7| 6| 5|
    |20|19|18|17| 4| 3| 2| 1|

    Parameters
    ----------
    raw    : [int]
    column : [int]

    Return
    ----------
    [int] address of the block in the mosaic given the raw and column identifier
    """
    raw_colomn_id = '(%i,%i)' % (raw, column)
    if raw_colomn_id == '(3,0)':
        return 0
    if raw_colomn_id == '(3,1)':
        return 1
    if raw_colomn_id == '(3,2)':
        return 2
    if raw_colomn_id == '(3,3)':
        return 3
    if raw_colomn_id == '(2,0)':
        return 4
    if raw_colomn_id == '(2,1)':
        return 5
    if raw_colomn_id == '(2,2)':
        return 6
    if raw_colomn_id == '(2,3)':
        return 7
    if raw_colomn_id == '(1,0)':
        return 8
    if raw_colomn_id == '(1,1)':
        return 9
    if raw_colomn_id == '(1,2)':
        return 10
    if raw_colomn_id == '(1,3)':
        return 11
    if raw_colomn_id == '(0,0)':
        return 12
    if raw_colomn_id == '(0,1)':
        return 13
    if raw_colomn_id == '(0,2)':
        return 14
    if raw_colomn_id == '(0,3)':
        return 15
    if raw_colomn_id == '(3,4)':
        return 16
    if raw_colomn_id == '(3,5)':
        return 17
    if raw_colomn_id == '(3,6)':
        return 18
    if raw_colomn_id == '(3,7)':
        return 19
    if raw_colomn_id == '(2,4)':
        return 20
    if raw_colomn_id == '(2,5)':
        return 21
    if raw_colomn_id == '(2,6)':
        return 22
    if raw_colomn_id == '(2,7)':
        return 23
    if raw_colomn_id == '(1,4)':
        return 24
    if raw_colomn_id == '(1,5)':
        return 25
    if raw_colomn_id == '(1,6)':
        return 26
    if raw_colomn_id == '(1,7)':
        return 27
    if raw_colomn_id == '(0,4)':
        return 28
    if raw_colomn_id == '(0,5)':
        return 29
    if raw_colomn_id == '(0,6)':
        return 30
    if raw_colomn_id == '(0,7)':
        return 31


def get_raw_image(filename):
    """
    Retrieve the mosaic from the fits file. The final list
    contains the 32 fits images.

    Parameters
    ----------
    filename : [str] name of the file

    Return
    ----------
    raw_images    : [numpy.array(np.float32)] unscaled mosaic
    header        : [astropy.io.fits.hdu.image.PrimaryHDU] header info
    """
    hdul = fits.open(filename)
    raw_images = []
    header = hdul[0].header
    for i in range(1, len(hdul)):
        raw_images.append(hdul[i].data)
    hdul.close()
    return raw_images[::-1], header


def compileVSTImages(path):
    # list of images
    filenames = []
    for file in os.listdir(path):
        if file.endswith(".fits"):
            filenames.append(os.path.join(
                path, file))
    for filename, k in zip(filenames, range(len(filenames))):
        hdul = fits.open(filename) # open image
        shape_one_tile = np.shape(hdul[1].data)
        raw_images, header = get_raw_image(filename)
        # create matrix to put in all the tiles together
        full_image = np.zeros((4*shape_one_tile[0], 8*shape_one_tile[1]))
        for i in range(4):
            for j in range(8):
                img = raw_images[map_blocks_adresses(i, j)]
                #crop_img, crop_results = post_results
                full_image[i*shape_one_tile[0]:(i+1)*shape_one_tile[0],
                        j*shape_one_tile[1]:(j+1)*shape_one_tile[1]] = img
        # save file to new fits image
        hdu = fits.PrimaryHDU(full_image)
        hdu.writeto('../VST/image{}.fits'.format(k), overwrite=True)
        print('../VST/image{}.fits'.format(k), ' saved')
    
if __name__ == '__main__':
    filename = '../VST/VST0.fits'
    img = fits.getdata(filename)
    bias = np.zeros(np.shape(img))
    dark = np.zeros(np.shape(img))
    flat = fits.getdata('../VST/skyflat0.fits')

    img_cal = calibrate(img, bias, dark, flat)
    hdu = fits.PrimaryHDU(img_cal)
    hdu.writeto(filename, overwrite=True)
