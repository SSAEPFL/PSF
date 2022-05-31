from astropy.stats import sigma_clipped_stats
from concurrent.futures.process import EXTRA_QUEUED_CALLS
from scipy.optimize import OptimizeWarning
import warnings
from ast import Try
import os
import numpy as np
from astropy.io import fits
import scipy.optimize as opt
from photutils.segmentation import make_source_mask
from scipy import stats
from scipy.signal import convolve2d
import numpy as np
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import zoom
plt.style.use(astropy_mpl_style)


warnings.simplefilter("error", OptimizeWarning)

def calibrate(img, bias, dark, flat):
    """ Image (img) calibration using a bias frame, a dark frame and a flat frame, the last three being average of multiple images
    : img: raw image of interest
      bias: bias image (shortest exposure of nothing) --> readout noise
      dark: dark image (same exposure time as img, but of pure darkness) --> thermal noise & quantum fluctuations
      flat: flat image (image to correct for vignetting)
      
      return: matrix of image
    """
    #clean_image = (img - dark) * np.mean(flat - dark) / (flat - dark) # source: https://en.wikipedia.org/wiki/Flat-field_correction
    flat = flat - bias
    # to avoid a zero, no difference in final result
    #flat = flat + (np.abs(np.min(flat))+1) * np.ones(shape=np.shape(flat))
    flat /= np.mean(flat)
    # source: http://spiff.rit.edu/classes/phys445/lectures/darkflat/darkflat.html
    clean_image = (img - bias) / flat
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

# The 2D Gaussian distribution function
def gaussian2D(xy, x0, y0, sigma_x, sigma_y, A=1, theta=0):
    """ Gaussian in 2D with maximum at (x0, y0) of amplitude A and std deviation (sigma_x, sigma_y) rotated around an angle theta
    : (x,y): position the function is evaluated at
      (x0, y0): center of gaussian
      (sigma_x, sigma_y): std deviation along both axes
      A: amplitude, if -1 then normalized to 1 (-1 by default)
      theta: angle of rotation (radian) (0 by default)
    return: scalar
    """
    (x, y) = np.asarray(xy).reshape(2, int(np.shape(xy)[0]/2))
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)**2/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    r = np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    if np.sum(r) != 0:
        return A*r/np.sum(r)
    else:
        return np.inf

def coordinatesOfStars(image):
    """ Get list of coordinates (i,j) of the sources in the image
    : image: matrix of an image
    
    return: list of coordinates
    """
    mask1 = make_source_mask(image, nsigma=7, npixels=10, dilate_size=1)
    mask2 = make_source_mask(image, nsigma=7, npixels=25, dilate_size=1)
    mask = np.logical_and(mask1, np.logical_not(mask2))
    i = 10
    list_of_coordinates = []
    inverted_masked_image = np.ma.array(image, mask=np.logical_not(mask), fill_value=np.nan)
    inverted_masked_image_filled = inverted_masked_image.filled()
    while i < (np.shape(mask)[0]-10):
        j = 10
        while j < (np.shape(mask)[1]-10):
            found_something = False
            if not (np.isnan(inverted_masked_image_filled[i-2:i+2, j-2:j+2])).any() and (i < 2100 or i > 2300 or j < 3000 or j > 3300):
                list_of_coordinates.append((i, j))
                found_something = True
                j += 7
                #print(inverted_masked_image[i:i+10, j:j+10])
            else:
                j += 1
        if found_something:
            i += 4
        else:
            i += 1  
    return list_of_coordinates

def fitForAllStars(image, list_of_coordinates):
    mask1 = make_source_mask(image, nsigma=7, npixels=10, dilate_size=1)
    mask2 = make_source_mask(image, nsigma=7, npixels=25, dilate_size=1)
    mask = np.logical_and(mask1, np.logical_not(mask2))
    mean, median, std = sigma_clipped_stats(image, sigma=4.0, mask=mask)
    parameters = []
    for coords in tqdm(list_of_coordinates, desc='stars', leave=False):
        i_begin = np.max([0, coords[0]-20])
        i_end = np.min([np.shape(image)[0], coords[0]+20])
        j_begin = np.max([0, coords[1]-20])
        j_end = np.min([np.shape(image)[1], coords[1]+20])

        outcut = image[i_begin:i_end, j_begin:j_end]
        #outcut = outcut/np.sum(outcut)

        initial_guess = (0, 0, 1, 1, 0.1, 0)
        len_i = i_end - i_begin
        len_j = j_end - j_begin
        xs = np.linspace(-int(len_i/2), int(len_i/2), np.shape(outcut)[1])
        ys = np.linspace(-int(len_j/2), int(len_j/2), np.shape(outcut)[0])
        xy = np.meshgrid(xs, ys)
        xy = np.ravel(xy)
        try:
            params, _ = opt.curve_fit(
                gaussian2D, xy, np.ravel(outcut), p0=initial_guess, bounds=([np.min(xs), np.min(ys), 0, 0, 0, -np.inf], [np.max(xs), np.max(ys), 20, 20, np.inf, np.inf]))
        except RuntimeError:
            params = [0, 0, 0, 0, 0]
        except OptimizeWarning:
            params = [0, 0, 0, 0, 0]
        if params[4]/std > 100 and np.abs(params[2]) > 0.1 and np.abs(params[3] > 0.1) and np.abs(params[2]) < 19.9 and np.abs(params[2]) < 19.9:
            parameters.append(params)
    return parameters

# fonctions to create and fit model of track
def trackModel(shape, i0, j0, iend, jend, width, amplitude):
    """ Creates a model of track in a frame of a given shape starting in position (i0, j0) and going in a straight line to (iend, jend) with a given width and amplitude
        shape: (lines, colunms), usually of np.shape()
        i0, j0, iend, jend: integer coordinates, inside of shape (including boundaries)
        width: scalar of width of track, measured perpendicular to the track
        amplitude: scalar, amplitude of track
    
    Return: outcut: matrix of size shape containing the track and zero outside
    """
    outcut = np.zeros(shape)  # create empty matrix
    L = np.sqrt((iend - i0)**2 + (jend - j0)**2)  # length of the track
    # angle of the track w.r.t. horizontal line
    alpha = np.arctan((iend - i0)/(jend - j0))
    # projection of the width along the vertical axis
    width_i = int(width/2 * np.cos(alpha))-1
    # projection of the width along the horizontal axis
    width_j = int(width/2 * np.sin(alpha))-1
    for t in range(int(np.floor(L))):  # along the track
        i = int(i0 + (iend - i0)/L * t)
        j = int(j0 + (jend - j0)/L * t)
        # for now: amplitude inside all the track +- width, maybe gaussian distribution ?
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            outcut[i, j] = amplitude
        else:
            raise RuntimeError('Begin or start point larger than matrix size')
        if width_i > 0 and (i - width_i) >= 0 and (i + width_i) < shape[0]:
            outcut[i-width_i:i+width_i, j] = amplitude
        if width_j > 0 and (j - width_j) >= 0 and (j + width_j) < shape[1]:
            outcut[i, j-width_j:j+width_j]
    return outcut

def simplerTrackModel(shape, i0, iend, width, amplitude):
    """ A simpler model of a track, where the start and end of the track is on the border """
    try:
       return trackModel(shape, i0, 0, iend, shape[1], width, amplitude)
    except RuntimeError:
        raise RuntimeError('Begin or start point larger than matrix size')


def reducedGoodnessModel(outcut, model, error):
    """ Measures goodness of track model compared to the outcut of the image containing the track, using chi2-test
        outcut: part of image containing the track
        model: track model
    Return: chi2: value of the chi2-test
    """
    if np.shape(outcut) != np.shape(model):
        raise ValueError('Not same shapes')
    if np.shape(outcut)[1] >= 1000:
        mask = make_source_mask(outcut, nsigma=4, npixels=np.shape(outcut)[1], dilate_size=2)
        outcut = np.ma.array(outcut, mask=np.logical_not(mask), fill_value=0)
    observed = np.ravel(outcut)
    expected = np.ravel(model)
    # simply chi^2 test, from https://arxiv.org/pdf/1012.3754.pdf
    chi2 = np.nansum(((observed - expected)/np.ravel(error))**2)
    if not np.isnan(chi2):
        return chi2/len(observed)
    else:
        print('NaN')
        return np.inf


def pvalue(outcut, model, error):
    """ P-value of the chi^2 test
        outcut: part of image containing the track, observed values
        model: part of image containing modelled track, expected values
    Return : p: p-value of the chi2 test
    """
    if np.shape(outcut) != np.shape(model):
        raise ValueError('Not same shapes')
    if np.shape(outcut)[1] >= 1000:
        mask = make_source_mask(outcut, nsigma=4, npixels=np.shape(outcut)[1], dilate_size=2)
        outcut = np.ma.array(outcut, mask=np.logical_not(mask), fill_value=0)
    observed = np.ravel(outcut)
    expected = np.ravel(model)
    # simply chi^2 test, from https://arxiv.org/pdf/1012.3754.pdf
    chi2 = np.nansum(((observed - expected)/np.ravel(error))**2)
    return stats.chi2.sf(chi2, len(observed)-4)


def logLikelihood(theta, y, yerr, PSF, shift=0, oversampling=False):
    """ Computes the log likelihood of the model specified by the parameters theta for the variables x (raveled coordinates of the outcut) and the measurements y (raveled outcut)
        theta: list of parameters (i0, j0, iend, jend, width, amplitude)
        x: indices of the raveled outcut model
        y: raveled outcut
    Return: -0.5*chi^2
    """
    shape = np.shape(y)
    if len(theta) == 4:
        i0, iend, width, amplitude = theta
        try:
            y_model = simplerTrackModel(shape, i0, iend, width, amplitude)
        except RuntimeError:
            return -np.inf
    elif len(theta) == 6:
        i0, j0, iend, jend, width, amplitude = theta
        try:
            y_model = trackModel(shape, i0, j0, iend, jend, width, amplitude)
        except RuntimeError:
            return -np.inf
    if oversampling:
        y = zoom(y, 2)  # oversampling fake data
        PSF = zoom(PSF, 2)
        y_model = zoom(y_model, 2)
    return -0.5*reducedGoodnessModel(y, convolve2d(y_model, PSF, 'same')+shift, yerr)


def logPrior(theta):
    """ Computes the prior probability of a given set of parameters theta (here assuming uniform distribution)
    return: scalar
    """
    if len(theta) == 6:
        i0, j0, iend, jend, width, amplitude = theta
        if 0 <= i0 < 500 and 0 <= iend < 500 and 0 <= j0 < 500 and 0 <= jend < 500 and 0 < width < 10 and 0 < amplitude < 65535:
            return 0  # log(probability) up to a constant
        else:
            return -np.inf
    elif len(theta) == 4:
        i0, iend, width, amplitude = theta
        if 0 <= i0 < 500 and 0 <= iend < 500 and 0 < width < 10 and 0 < amplitude < 65535:
            return 0  # log(probability) up to a constant
        else:
            return -np.inf
    else:
        raise ValueError('Not the right number of arguments to unpack in theta')


def logProbability(theta, y, yerr, PSF, shift=0, oversampling=False):
    lp = logPrior(theta)
    if np.isfinite(lp):
        return lp + logLikelihood(theta, y, yerr, PSF, shift, oversampling)
    return -np.inf


def folderPSF(path_to_folder, path_bias='bias', path_dark='dark', path_flats='flats'):
    """
    Computes the PSF of all stars in all images in folder, after having calibrated the images. Returns lists of sigma_x, sigma_y and theta
    """
    bias = averageFolder(path_bias)
    dark = averageFolder(path_dark)
    flat = averageFolder(path_flats)

    # get name of all files in folder
    image_names = []
    for file in os.listdir(path_to_folder):
        if file.endswith(".fit"):
            image_names.append(os.path.join(path_to_folder, file))

    # the values for each image
    sigma_x = []
    sigma_y = []
    theta = []
    A = []

    parameters = []
    for image_name in tqdm(image_names, desc='image', leave=False):
        img = fits.getdata(image_name)  # load image
        img = calibrate(img, bias, dark, flat)  # calibrate image
        list_of_coordinates = coordinatesOfStars(img)  # find all the stars in the image
        parameters += fitForAllStars(img, list_of_coordinates)

    sigma = [np.sqrt(x[2]**2 + x[3]) for x in parameters]        
    # sigma_x = [x[2] for x in parameters]
    # sigma_y = [x[3] for x in parameters]
    # theta = [x[5] for x in parameters]
    # A = [x[4] for x in parameters]
    return sigma


def loadParameters(angle_and_time):
    """
    Load the raw data obtained from the analysis of 
    """
    sigma = np.array(np.load(angle_and_time + '_sigma.npy', allow_pickle=True))
    return sigma
