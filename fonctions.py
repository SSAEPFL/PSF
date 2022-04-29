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
plt.style.use(astropy_mpl_style)

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
    flat = flat + (np.abs(np.min(flat))+1) * np.ones(shape=np.shape(flat))
    flat = flat / np.mean(flat)
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
    print(np.sum(r))
    return A*r/np.sum(r)

def coordinatesOfStars(image):
    """ Get list of coordinates (i,j) of the sources in the image
    : image: matrix of an image
    
    return: list of coordinates
    """
    mask = make_source_mask(image, nsigma=5, npixels=5, dilate_size=5)
    i = 5
    list_of_coordinates = []
    while i < (np.shape(mask)[0]-5):
        j = 5
        while j < (np.shape(mask)[1]-9):

            if mask[i, j-5:j+5].all():
                list_of_coordinates.append((i, j))
                j += 9
                #print(inverted_masked_image[i:i+10, j:j+10])
            else:
                j += 1
        i += 1
        
    return list_of_coordinates

def fitForAllStars(image, list_of_coordinates):
    parameters = []
    for coords in list_of_coordinates:
        i_begin = np.max([0, coords[0]-20])
        i_end = np.min([np.shape(image)[0], coords[0]+20])
        j_begin = np.max([0, coords[1]-20])
        j_end = np.min([np.shape(image)[1], coords[1]+20])

        outcut = image[i_begin:i_end, j_begin:j_end]
        outcut = outcut/np.sum(outcut)

        initial_guess = (0, 0, 1, 1, 0.1, 0)
        xs = np.linspace(-5, 5, np.shape(outcut)[1])
        ys = np.linspace(-5, 5, np.shape(outcut)[0])
        xy = np.meshgrid(xs, ys)
        xy = np.ravel(xy)
        try:
            params, _ = opt.curve_fit(
                gaussian2D, xy, np.ravel(outcut), p0=initial_guess)
        except RuntimeError:
            params = [0, 0, 0, 0, 0]
        parameters.append(params)
    #interpolated_data = gaussian2D(xy, params[0], params[1], params[2], params[3], params[4], params[5])
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
    width_i = int(width/2 * np.cos(alpha))
    # projection of the width along the horizontal axis
    width_j = int(width/2 * np.sin(alpha))
    for t in range(int(np.floor(L))):  # along the track
        i = int(i0 + (iend - i0)/L * t)
        j = int(j0 + (jend - j0)/L * t)
        # for now: amplitude inside all the track +- width, maybe gaussian distribution ?
        outcut[i, j] = amplitude
        if width_i > 0 and (i - width_i) >= 0 and (i + width_i) <= shape[0]:
            outcut[i-width_i:i+width_i, j] = amplitude
        if width_j > 0 and (j - width_j) >= 0 and (j + width_j) <= shape[1]:
            outcut[i, j-width_j:j+width_j]
    return outcut


def reducedGoodnessModel(outcut, model, error):
    """ Measures goodness of track model compared to the outcut of the image containing the track, using chi2-test
        outcut: part of image containing the track
        model: track model
    Return: chi2: value of the chi2-test
            p: p-value of the chi2-test, assuming track model 6 parameters
    """
    #return chisquare(np.ravel(outcut), np.ravel(model), 6)
    if np.shape(outcut) != np.shape(model):
        raise ValueError('Not same shapes')
    mask = make_source_mask(outcut, nsigma=4, npixels=1000, dilate_size=2)
    track = np.ma.array(outcut, mask=np.logical_not(mask), fill_value=0)
    measured = np.ravel(track.filled())
    simulated = np.ravel(model)
    # simply chi^2 test, from wikipedia
    chi2 = np.sum((simulated - measured)**2/np.ravel(error))
    p = stats.chi2.cdf(chi2, len(simulated)-6)  # assuming 6 parameters
    return chi2/len(simulated)

def logLikelihood(theta, y, yerr, PSF):
    """ Computes the log likelihood of the model specified by the parameters theta for the variables x (raveled coordinates of the outcut) and the measurements y (raveled outcut)
        theta: list of parameters (i0, j0, iend, jend, width, amplitude)
        x: indices of the raveled outcut model
        y: raveled outcut
    Return: -0.5*chi^2
    """
    shape = np.shape(y)
    i0, j0, iend, jend, width, amplitude = theta
    y_model = trackModel(shape, i0, j0, iend, jend, width, amplitude)
    return -0.5*reducedGoodnessModel(y, convolve2d(y_model, PSF, 'same'), yerr)


def logPrior(theta):
    """ Computes the prior probability of a given set of parameters theta (here assuming uniform distribution)
    return: scalar
    """
    i0, j0, iend, jend, width, amplitude = theta
    if 0 <= i0 <= 500 and 0 <= iend <= 500 and 0 <= j0 <= 500 and 0 <= jend <= 500 and 0 < width < 10 and 0 < amplitude < 65535:
        return 0 # log(probability) up to a constant
    else:
        return -np.inf


def logProbability(theta, y, yerr, PSF):
    lp = logPrior(theta)
    if np.isfinite(lp):
        return lp + logLikelihood(theta, y, yerr, PSF)
    return -np.inf