from astropy.io import fits
import numpy as np
import glob
import os

def subtract_value_from_fits(fits_file, value_to_subtract, output_file):
    """
    Subtracts a value from all pixels in a FITS file.
    If the result is negative, sets it to 1e-10.
    
    Parameters:
    -----------
    fits_file : str
        Path to the input FITS file
    value_to_subtract : float
        Value to subtract from each pixel
    output_file : str
        Path to save the modified file
    """
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Iterate over all extensions with image data
        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                # Subtract the value
                hdul[i].data = hdu.data - value_to_subtract
                
                # Replace negative values with 1e-10
                hdul[i].data[hdul[i].data < 0] = 1e-10
        
        # Save the file
        hdul.writeto(output_file, overwrite=True)
        print(f"Processed: {fits_file} -> {output_file}")

def process_folders(input_folders, output_folders, values_to_subtract):
    """
    Processes FITS files from multiple input folders to multiple output folders.
    
    Parameters:
    -----------
    input_folders : list of str
        List of input folder paths
    output_folders : list of str
        List of output folder paths (must match length of input_folders)
    values_to_subtract : list of float
        List of values to subtract (must match length of input_folders)
    """
    if len(input_folders) != len(output_folders) != len(values_to_subtract):
        raise ValueError("input_folders, output_folders, and values_to_subtract must have the same length")
    
    # Process each folder pair
    for input_folder, output_folder, value in zip(input_folders, output_folders, values_to_subtract):
        print(f"\n{'='*60}")
        print(f"Processing folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Value to subtract: {value}")
        print(f"{'='*60}")
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        # Find all .fits files in the input folder
        file_pattern = os.path.join(input_folder, '*.fits')
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"WARNING: No .fits files found in {input_folder}")
            continue
        
        print(f"Found {len(files)} files to process")
        
        # Process each file
        for file in files:
            # Get the filename without path
            filename = os.path.basename(file)
            
            # Create output file path
            output_file = os.path.join(output_folder, filename)
            
            # Process the file
            subtract_value_from_fits(file, value, output_file)
        
        print(f"Completed processing {input_folder}")
    
    print(f"\n{'='*60}")
    print("All folders processed successfully!")
    print(f"{'='*60}")

# Main execution
if __name__ == "__main__":
    # Define your 4 input folders
    input_folders = [
        'real_tidal_tails_crops_with_pedestal/VIS',
        'real_tidal_tails_crops_with_pedestal/NISP_H',
        'real_tidal_tails_crops_with_pedestal/NISP_J',
        'real_tidal_tails_crops_with_pedestal/NISP_Y'
    ]
    
    # Define your 4 output folders
    output_folders = [
        'real_tidal_tails_crops_without_pedestal/VIS',
        'real_tidal_tails_crops_without_pedestal/NISP_H',
        'real_tidal_tails_crops_without_pedestal/NISP_J',
        'real_tidal_tails_crops_without_pedestal/NISP_Y'
    ]
    
    # Define the 4 values to subtract (one for each folder)
    values_to_subtract = [
        0.023393100,
        64.125099,
        75.520447,
        57.394524
    ]
    
    # Process all folders
    process_folders(input_folders, output_folders, values_to_subtract)