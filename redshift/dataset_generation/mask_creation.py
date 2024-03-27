import glob
import copy
import re
import os
import cv2

import numpy as np
from astropy import fits

paths = ["make_mock_tidal_streams_VIS/", "make_mock_tidal_streams_NISP_H/", "make_mock_tidal_streams_NISP_J/", "make_mock_tidal_streams_NISP_Y/"]
paths_masks = ["segmentation_training/masks_train_VIS/", "segmentation_training/masks_train_NISP_H/", "segmentation_training/masks_train_NISP_J/", "segmentation_training/masks_train_NISP_Y/"]
size_VIS = 600
size_NISP = 200

for path_mask in paths_masks:
    if not os.path.exists(path_mask):
        os.makedirs(path_mask)


def get_galaxy_number(galaxy_name:str):
    return galaxy_name.split('.')[0].split('_')[-2] 

def get_galaxy_magnitude(galaxy_name:str):
    pattern = re.compile(r'\d+\.\d+')
    return pattern.findall(galaxy_name)[0]

def create_masks():
    for path,path_masks in zip(paths, paths_masks):
        for file in glob.glob(path+"stream_pix_above_surf_bright_limit_*"):
            print("Creating mask for file: "+file)
            galaxy_number = get_galaxy_number(file)
            galaxy_magnitude = get_galaxy_magnitude(file)
            with fits.open(file) as hdul:
                fits_data = hdul[1].data
                if path == "make_mock_tidal_streams_VIS/":
                    stream_mask = np.zeros((size_VIS, size_VIS))
                else:
                    stream_mask = np.zeros((size_NISP, size_NISP))
                stream_mask[fits_data!=0]=1
                cv2.imwrite(path_masks+"mask_"+str(galaxy_number)+"_"+str(galaxy_magnitude)+".png", stream_mask)