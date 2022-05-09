import os
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
plt.style.use(astropy_mpl_style)


if True:
    # Prepare for calibration
    from fonctions import *
    bias = averageFolder('bias')
    dark = averageFolder('dark')
    flat = averageFolder('../flats')

    # get name of all files in folder
    path_to_folder = "../working-dir"
    image_names = []
    for file in os.listdir(path_to_folder):
        if file.endswith(".fit"):
            image_names.append(os.path.join(path_to_folder, file))
    print('{} images to analyse'.format(len(image_names)))
        
    # the values for each image
    sigma_x = []
    sigma_y = []
    theta = []
    
    parameters = []
    for image_name in image_names:
        print('\tImage name : ', image_name)
        img = fits.getdata(image_name) # load image
        img = calibrate(img, bias, dark, flat) # calibrate image
        list_of_coordinates = coordinatesOfStars(img) # find all the stars in the image
        print('{} stars to analyse'.format(len(list_of_coordinates)))
        n_old, n = 0, 1000
        while n < len(list_of_coordinates):
            print('\tn_old and n : ', n_old, n)
            parameters = fitForAllStars(img, list_of_coordinates[n_old:n-1])
            n_old = n
            n += 1000
        if n > len(list_of_coordinates):
            parameters = fitForAllStars(img, list_of_coordinates[n_old:])  
            
    print('Parameters[10]', parameters[10])     
    print('len parameters', len(parameters))
            
    sigma_x = [x[2] for x in parameters]
    sigma_y = [x[3] for x in parameters]
    theta = [x[4] for x in parameters]


    # save the raw data to avoid running the code againg
    with open("20deg0.5s_sigma_x.npy", "wb") as file:
        np.save(file, np.asarray(sigma_x))

    with open("20deg0.5s_sigma_y.npy", "wb") as file:
        np.save(file, np.asarray(sigma_y))

    with open("20deg0.5s_theta.npy", "wb") as file:
        np.save(file, np.asarray(theta))
        
if False:
    # load the raw data to avoid running the code againg
    sigma_x = np.array(np.load('20deg0.5s_sigma_x.npy', allow_pickle=True))
    sigma_y = np.array(np.load('20deg0.5s_sigma_y.npy', allow_pickle=True))
    theta = np.array(np.load('20deg0.5s_theta.npy', allow_pickle=True))

sigma_x = [x for x in sigma_x if x > 1 and x < 100]
sigma_y = [x for x in sigma_y if x > 1 and x < 100]
e = [x/y for x,y in zip(sigma_x, sigma_y)]
e = [1/x for x in e if x > 1] 


    
# compute the interesting values and plot them
sigma_x_avg = np.mean(sigma_x)
sigma_x_med = np.median(sigma_x)
sigma_x_std = np.std(sigma_x)

plt.figure()
plt.hist(sigma_x, bins=64)
plt.vlines(sigma_x_avg, ymin=0, ymax=100, label='mean', color='red')
plt.vlines(sigma_x_med, ymin=0, ymax=100, label='median', color='red')
plt.vlines([sigma_x_avg - sigma_x_std, sigma_x_avg + sigma_x_std], label=r'$1\sigma$', ymin=0, ymax=100, color='black')
plt.xlabel(r'$\sigma_x$ [px]')
plt.ylabel(r'Number of appearance')
plt.show()

sigma_y_avg = np.mean(sigma_y)
sigma_y_med = np.median(sigma_y)
sigma_y_std = np.std(sigma_y)

plt.figure()
plt.hist(sigma_y, bins=64)
plt.vlines(sigma_y_avg, ymin=0, ymax=100, label='mean', color='red')
plt.vlines(sigma_y_med, ymin=0, ymax=100, label='median', color='red')
plt.vlines([sigma_y_avg - sigma_y_std, sigma_y_avg + sigma_y_std], label=r'$1\sigma$', ymin=0, ymax=100, color='black')
plt.xlabel(r'$\sigma_y$ [px]')
plt.ylabel(r'Number of appearance')
plt.show()

theta_avg = np.mean(theta)
theta_med = np.median(theta)
theta_std = np.std(theta)

plt.figure()
plt.hist(theta, bins=64)
plt.vlines(theta_avg, ymin=0, ymax=100, label='mean', color='red')
plt.vlines(theta_med, ymin=0, ymax=100, label='median', color='red')
plt.vlines([theta_avg - theta_std, theta_avg + theta_std], label=r'$1\sigma$', ymin=0, ymax=100, color='black')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'Number of appearance')
plt.show()

plt.figure()
plt.hist(e, bins=64)
plt.xlabel(r'$e$')
plt.ylabel(r'Number of appearance')
plt.show()
