#!/usr/bin/env python3
"""
Script to generate segmentation of FITS files using SExtractor
Processes all .fits files from a specified folder
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

def run_sextractor_segmentation(input_file, output_dir, config_file="sx_config_file.sex", config_params=None):
    """
    Runs SExtractor segmentation on a FITS file
    
    Args:
        input_file: Path to input FITS file
        output_dir: Output directory
        config_file: SExtractor configuration file
        config_params: Dictionary with additional parameters
    """
    
    # Base filename without extension
    base_name = Path(input_file).stem
    
    # Output segmentation file
    segmentation_file = os.path.join(output_dir, f"{base_name}_segmentation.fits")
    
    # Build SExtractor command
    sex_cmd = [
        'sex',
        input_file,
        '-c', config_file,
        '-CHECKIMAGE_TYPE', 'SEGMENTATION',
        '-CHECKIMAGE_NAME', segmentation_file
    ]
    
    # Add additional parameters if provided
    if config_params:
        for param, value in config_params.items():
            sex_cmd.extend([param, str(value)])
    
    try:
        print(f"Processing: {input_file}")
        
        # Run SExtractor
        print("  - Running SExtractor...")
        result = subprocess.run(sex_cmd, capture_output=True, text=True, check=True)
        
        print(f"  ✓ Completed: {segmentation_file}")
        
        return {
            'segmentation': segmentation_file,
            'success': True,
            'input_file': input_file
        }
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error processing {input_file}:")
        print(f"    {e.stderr}")
        return {'success': False, 'error': str(e), 'input_file': input_file}

def process_fits_directory(input_dir, output_dir=None, config_file="sx_config_file.sex", config_params=None):
    """
    Processes all FITS files in a directory
    
    Args:
        input_dir: Directory with FITS files
        output_dir: Output directory (if None, uses input_dir/segmentation)
        config_file: SExtractor configuration file path
        config_params: Custom parameters for SExtractor
    """
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'segmentation')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for FITS files
    fits_pattern = os.path.join(input_dir, "*.fits")
    fits_files = glob.glob(fits_pattern)
    
    if not fits_files:
        print(f"No .fits files found in {input_dir}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {config_file}")
    print("-" * 50)
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    
    for fits_file in fits_files:
        result = run_sextractor_segmentation(fits_file, output_dir, config_file, config_params)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print("-" * 50)
    print(f"Processing completed:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(fits_files)}")
    
    return results

def run_sextractor_no_config(input_file, output_dir, config_params=None):
    """
    Runs SExtractor without config file (using defaults)
    
    Args:
        input_file: Path to input FITS file
        output_dir: Output directory
        config_params: Dictionary with additional parameters
    """
    
    # Base filename without extension
    base_name = Path(input_file).stem
    
    # Output segmentation file
    segmentation_file = os.path.join(output_dir, f"{base_name}_segmentation.fits")
    
    # Build SExtractor command without config file
    sex_cmd = [
        'sex',
        input_file,
        '-CHECKIMAGE_TYPE', 'SEGMENTATION',
        '-CHECKIMAGE_NAME', segmentation_file
    ]
    
    # Add additional parameters if provided
    if config_params:
        for param, value in config_params.items():
            sex_cmd.extend([param, str(value)])
    
    try:
        print(f"Processing: {input_file}")
        
        # Run SExtractor
        print("  - Running SExtractor (no config file)...")
        result = subprocess.run(sex_cmd, capture_output=True, text=True, check=True)
        
        print(f"  ✓ Completed: {segmentation_file}")
        
        return {
            'segmentation': segmentation_file,
            'success': True,
            'input_file': input_file
        }
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error processing {input_file}:")
        print(f"    {e.stderr}")
        return {'success': False, 'error': str(e), 'input_file': input_file}


def main():
    """Main script function"""
    
    # Configuration - MODIFY ACCORDING TO YOUR NEEDS
    # Define pairs of (input_directory, output_directory)
    directory_pairs = [
        ("../../segmentation_training/v13_log/galaxies_train_VIS_with_pedestal", 
         "../../segmentation_training/v13_log/galaxies_train_VIS_segmented_sextractor"),
        ("../../segmentation_training/v13_log/galaxies_train_NISP_H_with_pedestal", 
         "../../segmentation_training/v13_log/galaxies_train_NISP_H_segmented_sextractor"),
        ("../../segmentation_training/v13_log/galaxies_train_NISP_J_with_pedestal", 
         "../../segmentation_training/v13_log/galaxies_train_NISP_J_segmented_sextractor"),
        ("../../segmentation_training/v13_log/galaxies_train_NISP_Y_with_pedestal", 
         "../../segmentation_training/v13_log/galaxies_train_NISP_Y_segmented_sextractor"),
        # Add more pairs as needed
    ]
    
    config_file = "sx_config_file.sex"  # SExtractor config file
    
    # Check command line arguments for override
    if len(sys.argv) > 1:
        print("Command line arguments detected. Using single directory mode.")
        input_directory = sys.argv[1]
        output_directory = sys.argv[2] if len(sys.argv) > 2 else None
        config_file = sys.argv[3] if len(sys.argv) > 3 else config_file
        
        directory_pairs = [(input_directory, output_directory)]
    
    print("=" * 60)
    print("FITS FILES SEGMENTATION WITH SEXTRACTOR - MULTIPLE DIRECTORIES")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Processing {len(directory_pairs)} directory pairs:")
    print()
    
    # Process each directory pair
    all_results = []
    for i, (input_dir, output_dir) in enumerate(directory_pairs, 1):
        print(f"Processing pair {i}/{len(directory_pairs)}:")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print()
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory does not exist: {input_dir}")
            print("Skipping this pair...\n")
            continue
            
        try:
            # Run processing for this pair
            results = process_fits_directory(
                input_dir, 
                output_dir, 
                config_file,
            )
            all_results.append((input_dir, output_dir, results))
            print(f"Completed processing pair {i}")
            print("-" * 40)
            print()
            
        except Exception as e:
            print(f"Error processing pair {i}: {str(e)}")
            print("Continuing with next pair...\n")
            all_results.append((input_dir, output_dir, None))
    
    # Summary
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, _, result in all_results if result is not None)
    print(f"Successfully processed: {successful}/{len(directory_pairs)} pairs")
    
    for i, (input_dir, output_dir, result) in enumerate(all_results, 1):
        status = "✓ Success" if result is not None else "✗ Failed"
        print(f"Pair {i}: {status}")
        print(f"  {input_dir} -> {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()