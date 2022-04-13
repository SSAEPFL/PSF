from fonctions import *
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)

# Prepare for calibration
from fonctions import *
bias = fits.getdata('bias.fit')
dark = fits.getdata('dark.fit')
flat = fits.getdata('flat.fit')

bias = averageFolder('bias')
dark = averageFolder('dark')
flat = averageFolder('flat')

# Read image, calibrate and measure PSF
sigma_x_map = np.zeros(np.shape(flat))
sigma_y_map = np.zeros(np.shape(flat))
theta_map = np.zeros(np.shape(flat))

# , 'image2.fit', 'image3.fit', 'image4.fit', 'image5.fit', 'image6.fit', 'image7.fit', 'image8.fit']
image_names = ['image4.fit', 'image6.fit', 'image7.fit', 'image8.fit']

for image_name in image_names:
    print('\tImage name : ', image_name)
    img = fits.getdata(image_name)
    img = calibrate(img, bias, dark, flat)
    list_of_coordinates = coordinatesOfStars(img)
    parameters = []
    n_old, n = 0, 1000
    while n < len(list_of_coordinates):
        print('\tn_old and n : ', n_old, n)
        parameters += fitForAllStars(img, list_of_coordinates[n_old:n-1])
        n_old = n
        n += 1000
    if n > len(list_of_coordinates):
        parameters += fitForAllStars(img, list_of_coordinates[n_old:])                        
    
    for i in range(len(parameters)):
        coords = list_of_coordinates[i]
        i_begin = np.max([0, coords[0]-20])
        i_end = np.min([np.shape(img)[0], coords[0]+20])
        j_begin = np.max([0, coords[1]-20])
        j_end = np.min([np.shape(img)[1], coords[1]+20])
        sigma_x_map[i_begin:i_end, j_begin:j_end] += parameters[i][2]/len(image_names)
        sigma_y_map[i_begin:i_end, j_begin:j_end] += parameters[i][3]/len(image_names)
        theta_map[i_begin:i_end, j_begin:j_end] += parameters[i][4]/len(image_names)
        
textfile = open("sigma_x_map.dat", "w")
sigma_x_map.tofile(textfile)
textfile.close()

textfile = open("sigma_y_map.dat", "w")
sigma_y_map.tofile(textfile)
textfile.close()

textfile = open("theta_map.dat", "w")
theta_map.tofile(textfile)
textfile.close()
        
plt.figure()
plt.title(r'$\sigma_x$')
plt.imshow(np.abs(sigma_x_map), origin='lower',
           cmap='viridis', interpolation='none')
plt.colorbar()
plt.show()

plt.figure()
plt.title(r'$\sigma_y$')
plt.imshow(np.abs(sigma_y_map), origin='lower',
           cmap='viridis', interpolation='none')
plt.colorbar()
plt.show()

plt.figure()
plt.title(r'$\theta$')
plt.imshow(theta_map*180/np.pi, origin='lower',
           cmap='viridis', interpolation='none')
plt.colorbar()
plt.show()
