from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import numpy as np
from fonctions import *
plt.style.use(astropy_mpl_style)
from multiprocessing import Pool

def treatFolder(folder, master_dir="../images-sat/"):
    sigma_x, sigma_y, theta, A = folderPSF(master_dir + folder, path_flats='flat')
    # save the raw data to avoid running the code againg
    sigma_x = [x for x in sigma_x if x > 1]
    sigma_x = np.argpartition(sigma_x, 20)
    
    sigma_y = [x for x in sigma_y if x > 1]
    sigma_y = np.argpartition(sigma_y, 20)
    
    theta = [x for x in theta if x > 1]
    theta = np.argpartition(theta, 20)
    
    
    with open('data/' + folder + "_sigma_x.npy", "wb") as file:
        np.save(file, np.asarray(sigma_x))
    with open('data/' + folder + "_sigma_y.npy", "wb") as file:
        np.save(file, np.asarray(sigma_y))
    with open('data/' + folder + "_theta.npy", "wb") as file:
        np.save(file, np.asarray(theta))


if __name__ == "__main__":
    master_dir = "../images-sat/"  # path to folder containing folders
    folder_names = []  # list of path to folders
    for f in os.listdir(master_dir):
        if os.path.isdir(master_dir + f):
            folder_names.append(f)       
    print(folder_names)

    #for folder in tqdm(folder_names):
    #    treatFolder(folder)
    pool = Pool()
    pool.map(treatFolder, folder_names)
    

    # load the parameters
    sigma_x, sigma_y, theta = loadParameters('data/'+folder_names[0])

    # clean the data
    sigma_x = [x for x in sigma_x if x > 1 and x < 100]
    sigma_y = [x for x in sigma_y if x > 1 and x < 100]
    e = [x/y for x, y in zip(sigma_x, sigma_y)]
    e = [1/x for x in e if x > 1]

    # compute the interesting values...
    sigma_x_avg = np.mean(sigma_x)
    sigma_x_med = np.median(sigma_x)
    sigma_x_std = np.std(sigma_x)

    # ...and plot them
    plt.figure()
    plt.hist(sigma_x, bins=64)
    plt.vlines(sigma_x_avg, ymin=0, ymax=100, label='mean', color='red')
    plt.vlines(sigma_x_med, ymin=0, ymax=100, label='median', color='red')
    plt.vlines([sigma_x_avg - sigma_x_std, sigma_x_avg + sigma_x_std],
               label=r'$1\sigma$', ymin=0, ymax=100, color='black')
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
    plt.vlines([sigma_y_avg - sigma_y_std, sigma_y_avg + sigma_y_std],
               label=r'$1\sigma$', ymin=0, ymax=100, color='black')
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
    plt.vlines([theta_avg - theta_std, theta_avg + theta_std],
               label=r'$1\sigma$',  ymin=0, ymax=100, color='black')
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'Number of appearance')
    plt.show()

    plt.figure()
    plt.hist(e, bins=64)
    plt.xlabel(r'$e$')
    plt.ylabel(r'Number of appearance')
    plt.show()
