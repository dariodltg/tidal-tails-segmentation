from astropy.io import fits
import numpy as np
import glob
import copy
import os

paths = [
         "../segmentation_training/v4/galaxies_train_VIS/","../segmentation_training/v4/galaxies_train_NISP_H/","../segmentation_training/v4/galaxies_train_NISP_J/","../segmentation_training/v4/galaxies_train_NISP_Y/",
         "../segmentation_training/v4/galaxies_test_VIS/","../segmentation_training/v4/galaxies_test_NISP_H/","../segmentation_training/v4/galaxies_test_NISP_J/","../segmentation_training/v4/galaxies_test_NISP_Y/"]


def delete_nans():
    for path in paths:
        print("Checking nans and infs: " + path)
        for file in glob.glob(path+"galaxy_and_stream_convolved*"):
            with fits.open(file) as hdul:  # Open fits file
                data = hdul[0].data.astype(np.float32)
                if(np.isnan(data).any()):
                    print("Nans in file: "+file)
                    os.remove(file)
                if(np.isinf(data).any()):
                    print("Infs in file: "+file)
            