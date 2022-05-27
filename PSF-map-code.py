from fonctions import *
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)
from fonctions import *

bias = averageFolder('bias')
dark = averageFolder('dark')
flat = averageFolder('flat')

# Read image, calibrate and measure PSF
sigma_x_map = np.zeros(np.shape(flat))
sigma_y_map = np.zeros(np.shape(flat))
theta_map = np.zeros(np.shape(flat))

# getting the names of all images
image_names = []
for file in os.listdir('../psf-map-images/'):
    if file.endswith(".fit"):
        image_names.append(os.path.join('../psf-map-images/', file))

# for each image, find all the stars, fit each star and add the parameters to the map
for image_name in tqdm(image_names, desc='Image names', leave=True):
    img = fits.getdata(image_name)
    img = calibrate(img, bias, dark, flat)
    list_of_coordinates = coordinatesOfStars(img)
    parameters = fitForAllStars(img, list_of_coordinates)                        
    for i in range(len(parameters)):
        coords = list_of_coordinates[i]
        i_begin = np.max([0, coords[0]-20])
        i_end = np.min([np.shape(img)[0], coords[0]+20])
        j_begin = np.max([0, coords[1]-20])
        j_end = np.min([np.shape(img)[1], coords[1]+20])
        if parameters[2] < 30 and parameters[3] < 30:
            sigma_x_map[i_begin:i_end, j_begin:j_end] = parameters[i][2]
            sigma_y_map[i_begin:i_end, j_begin:j_end] = parameters[i][3]
            theta_map[i_begin:i_end, j_begin:j_end] = parameters[i][4]
   
# saving the maps     
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
