"""
FITS image cropping for VIS and NISP filters.
Crops VIS at given coordinates, and NISP (H, J, Y) at scaled coordinates (1/3 resolution).
Uses montage_wrapper for cropping while preserving WCS.
"""

import montage_wrapper as montage
import os
from astropy.io import fits


# =============================================================================
# PARAMETERS - EDIT HERE
# =============================================================================

# Input images
input_VIS = "lsb_optimized_jan25/EUC_VIS_LSB_ObsID_2706.fits"
input_NISP_H = "lsb_optimized_jan25/EUC_NIR_H_LSB_ObsID_2706.fits"
input_NISP_J = "lsb_optimized_jan25/EUC_NIR_J_LSB_ObsID_2706.fits"
input_NISP_Y = "lsb_optimized_jan25/EUC_NIR_Y_LSB_ObsID_2706.fits"

# Crop center coordinates (x, y) - in VIS pixel scale
x_center = 4960
y_center = 28347

# Crop sizes (pixels)
size_VIS = 600
size_NISP = 200  # VIS/3

# Output folders (created if they don't exist)
output_folder_VIS = "real_tidal_tails_crops/VIS/"
output_folder_NISP_H = "real_tidal_tails_crops/NISP_H/"
output_folder_NISP_J = "real_tidal_tails_crops/NISP_J/"
output_folder_NISP_Y = "real_tidal_tails_crops/NISP_Y/"

# HDU to crop (0 for PrimaryHDU)
hdu_index = 0


# =============================================================================
# FUNCTIONS
# =============================================================================

def crop_fits_at_coordinates(input_path, x, y, size, output_path, hdu=0):
    """
    Crop a FITS image centered on coordinates (x, y).
    
    Parameters
    ----------
    input_path : str
        Path to the input FITS file.
    x : int
        X coordinate of the crop center.
    y : int
        Y coordinate of the crop center.
    size : int
        Crop size (size x size pixels).
    output_path : str
        Output path for the crop.
    hdu : int
        HDU index to crop.
    
    Returns
    -------
    str
        Path to the generated file.
    """
    # Calculate lower-left corner
    pix_halfsize = int(size / 2)
    x_start = x - pix_halfsize
    y_start = y - pix_halfsize
    
    # Get original image info
    with fits.open(input_path) as hdu_list:
        original_shape = hdu_list[hdu].data.shape
        print(f"Original image: {input_path}")
        print(f"Dimensions: {original_shape}")
        print(f"Crop center: ({x}, {y})")
        print(f"Crop size: {size}x{size}")
    
    # Check that crop is within bounds
    if x_start < 0 or y_start < 0:
        raise ValueError(f"Crop out of bounds: start ({x_start}, {y_start}) < 0")
    if x_start + size > original_shape[1] or y_start + size > original_shape[0]:
        raise ValueError(f"Crop out of bounds: exceeds image dimensions")
    
    # Perform crop with montage
    montage.mSubimage_pix(
        input_path,
        output_path,
        x_start,
        y_start,
        hdu=hdu,
        xpixsize=size,
        ypixsize=size
    )
    
    # Add crop metadata to header
    with fits.open(output_path, mode='update') as hdu_out:
        hdu_out[0].header['HISTORY'] = f'Crop centered at ({x}, {y}), size {size}x{size}'
        hdu_out[0].header['CROP_X'] = (x, 'Crop center X')
        hdu_out[0].header['CROP_Y'] = (y, 'Crop center Y')
        hdu_out[0].header['CROP_SZ'] = (size, 'Crop size')
        hdu_out[0].header['ORIG_IMG'] = (os.path.basename(input_path), 'Original image')
        hdu_out.flush()
    
    print(f"Crop saved: {output_path}")
    
    return output_path


def create_output_folders():
    """Create all output folders if they don't exist."""
    folders = [output_folder_VIS, output_folder_NISP_H, output_folder_NISP_J, output_folder_NISP_Y]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    
    # Create output folders
    create_output_folders()
    
    # Scale coordinates for NISP (1/3 resolution)
    x_center_nisp = int(x_center / 3)
    y_center_nisp = int(y_center / 3)
    
    # Output filename with both VIS and NISP coordinates for easy identification
    output_name = f"VIS_{x_center}_{y_center}_NISP_{x_center_nisp}_{y_center_nisp}.fits"
    
    # Crop VIS image
    print("\n" + "="*50)
    print("Cropping VIS image")
    print("="*50)
    crop_fits_at_coordinates(
        input_path=input_VIS,
        x=x_center,
        y=y_center,
        size=size_VIS,
        output_path=os.path.join(output_folder_VIS, output_name),
        hdu=hdu_index
    )
    
    # Crop NISP H image
    print("\n" + "="*50)
    print("Cropping NISP H image")
    print("="*50)
    crop_fits_at_coordinates(
        input_path=input_NISP_H,
        x=x_center_nisp,
        y=y_center_nisp,
        size=size_NISP,
        output_path=os.path.join(output_folder_NISP_H, output_name),
        hdu=hdu_index
    )
    
    # Crop NISP J image
    print("\n" + "="*50)
    print("Cropping NISP J image")
    print("="*50)
    crop_fits_at_coordinates(
        input_path=input_NISP_J,
        x=x_center_nisp,
        y=y_center_nisp,
        size=size_NISP,
        output_path=os.path.join(output_folder_NISP_J, output_name),
        hdu=hdu_index
    )
    
    # Crop NISP Y image
    print("\n" + "="*50)
    print("Cropping NISP Y image")
    print("="*50)
    crop_fits_at_coordinates(
        input_path=input_NISP_Y,
        x=x_center_nisp,
        y=y_center_nisp,
        size=size_NISP,
        output_path=os.path.join(output_folder_NISP_Y, output_name),
        hdu=hdu_index
    )
    
    print("\n" + "="*50)
    print("All crops completed!")
    print(f"VIS coordinates: ({x_center}, {y_center})")
    print(f"NISP coordinates: ({x_center_nisp}, {y_center_nisp})")
    print("="*50)
