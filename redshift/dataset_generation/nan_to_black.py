from astropy.io import fits
import numpy as np
import glob
import copy
import os

paths = ["make_mock_tidal_streams_VIS/", "make_mock_tidal_streams_NISP_H/", "make_mock_tidal_streams_NISP_J/", "make_mock_tidal_streams_NISP_Y/"]

def nan_to_black():
    for path in paths:
        print("Processing nan pixels to black in path" + path)
        for file in glob.glob(path+"galaxy_and_stream_convolved*"):
            with fits.open(file, mode="update") as hdul:  # Open fits file
                data = hdul[1].data
                data[np.isnan(data)] = 0  # Replace nans with zeros
                hdul.flush()  # Save changes
