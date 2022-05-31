from astropy.visualization import astropy_mpl_style
import matplotlib.pyplot as plt
import numpy as np
from fonctions import *
plt.style.use(astropy_mpl_style)
from multiprocessing import Pool

def treatFolder(folder, master_dir="../images-sat/"):
    sigma = folderPSF(master_dir + folder, path_flats='flat')
    # save the raw data to avoid running the code againg    
    
    with open('data/' + folder + "_sigma.npy", "wb") as file:
        np.save(file, np.asarray(sigma))


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
    sigma = loadParameters('data/'+folder_names[0])

    # clean the data
    sigma = [x for x in sigma if x > 1 and x < 100]

    # compute the interesting values...
    sigma_avg = np.mean(sigma)
    sigma_med = np.median(sigma)
    sigma_std = np.std(sigma)

    # ...and plot them
    plt.figure()
    plt.hist(sigma, bins=64)
    plt.vlines(sigma_avg, ymin=0, ymax=100, label='mean', color='red')
    plt.vlines(sigma_med, ymin=0, ymax=100, label='median', color='red')
    plt.vlines([sigma_avg - sigma_std, sigma_avg + sigma_std],
               label=r'$1\sigma$', ymin=0, ymax=100, color='black')
    plt.xlabel(r'$\sigma$ [px]')
    plt.ylabel(r'Number of appearance')
    plt.show()
