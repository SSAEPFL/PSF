import os
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import statistics as stats
plt.style.use(astropy_mpl_style)

# Prepare for calibration
from fonctions import *
bias = averageFolder('bias')
dark = averageFolder('dark')
flat = averageFolder('flat')

# get name of all files in folder
path_to_folder = "20deg"
image_names = []
for file in os.listdir(path_to_folder):
    if file.endswith(".fit"):
        image_names.append(os.path.join(path_to_folder, file))
        
# the values for each image
sigma_x = []
sigma_y = []
theta = []

for image_name in image_names:
    print('\tImage name : ', image_name)
    img = fits.getdata(image_name) # load image
    img = calibrate(img, bias, dark, flat) # calibrate image
    list_of_coordinates = coordinatesOfStars(img) # find all the stars in the image
    parameters = []
    n_old, n = 0, 1000
    while n < len(list_of_coordinates):
        print('\tn_old and n : ', n_old, n)
        parameters = fitForAllStars(img, list_of_coordinates[n_old:n-1])
        sigma_x.append(parameters[2])
        sigma_y.append(parameters[3])
        theta.append(parameters[4])
        n_old = n
        n += 1000
    if n > len(list_of_coordinates):
        parameters = fitForAllStars(img, list_of_coordinates[n_old:])                        

# save the raw data to avoid running the code againg
textfile = open("angle_sigma_x.dat", "w")
sigma_x.tofile(textfile)
textfile.close()

textfile = open("angle_sigma_y.dat", "w")
sigma_y.tofile(textfile)
textfile.close()

textfile = open("angle_theta.dat", "w")
theta.tofile(textfile)
textfile.close()

# compute the interesting values and plot them
sigma_x_avg = np.mean(sigma_x)
sigma_x_med = np.median(sigma_x)
sigma_x_std = stats.stdev(sigma_x)

plt.figure()
plt.hist(sigma_x, bins=100)
plt.vlines(sigma_x_avg, label='mean')
plt.vlines(sigma_x_med, label='median')
plt.vlines([sigma_x_avg - sigma_x_std, sigma_x_avg + sigma_x_std], label=r'$1\sigma$')
plt.xlabel(r'$\sigma_x$ [px]')
plt.ylabel(r'Number of appearance')
plt.show()

sigma_y_avg = np.mean(sigma_y)
sigma_y_med = np.median(sigma_y)
sigma_y_std = stats.stdev(sigma_y)

plt.figure()
plt.hist(sigma_y, bins=100)
plt.vlines(sigma_y_avg, label='mean')
plt.vlines(sigma_y_med, label='median')
plt.vlines([sigma_y_avg - sigma_y_std, sigma_y_avg + sigma_y_std], label=r'$1\sigma$')
plt.xlabel(r'$\sigma_y$ [px]')
plt.ylabel(r'Number of appearance')
plt.show()


theta_avg = np.mean(theta)
theta_med = np.median(theta)
theta_std = stats.stdev(theta)

plt.figure()
plt.hist(theta, bins=100)
plt.vlines(theta_avg, label='mean')
plt.vlines(theta_med, label='median')
plt.vlines([theta_avg - theta_std, theta_avg + theta_std], label=r'$1\sigma$')
plt.xlabel(r'$\theta$ [px]')
plt.ylabel(r'Number of appearance')
plt.show()
