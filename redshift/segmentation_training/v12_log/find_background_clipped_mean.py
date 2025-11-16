import glob
from astropy.io import fits
import numpy as np
import pandas as pd
import montage_wrapper as montage
import os.path
import pdb
import matplotlib.pyplot as plt
import sys
import scipy.stats as stt
import os
import glob
import numpy as np
from astropy.io import fits

def resistant_mean(vector, threshold = 5.):
    """
    this function is (numerically) the same as the resistant_mean function in IDL
    """
    vector = vector[~np.isnan(vector)]
    if vector.size == 0:
        return(float("NaN"),float("NaN"),float("NaN"))
    else:
        clipping_apers                         = stt.sigmaclip(vector,low=threshold,high=threshold)
        objs_after_clipping                    = clipping_apers[0] #objects after trimming
        mean_objs_after_clipping               = np.nanmean(objs_after_clipping)
        stddev_objs_after_clipping             = np.nanstd(objs_after_clipping) #this is the scatter or the standard deviation
        stddev_of_the_mean_objs_after_clipping = stddev_objs_after_clipping/np.sqrt(objs_after_clipping.size-1) #this is the standard deviation of the mean
        return(mean_objs_after_clipping,stddev_objs_after_clipping,stddev_of_the_mean_objs_after_clipping)

def calculate_background_mean(image_path, segm_path):
    """Calculate background mean for a single image pair using resistant_mean"""
    try:
        # Load images
        with fits.open(image_path) as hdul:
            image_data = hdul[0].data
        
        with fits.open(segm_path) as hdul:
            segm_data = hdul[0].data
        
        # Check dimensions
        if image_data.shape != segm_data.shape:
            print(f"Error: Shape mismatch - Image: {image_data.shape}, Segmentation: {segm_data.shape}")
            return None
        
        # Create background mask (segmentation == 0)
        background_mask = (segm_data == 0)
        
        # Extract background pixels
        background_pixels = image_data[background_mask]
        
        # Remove NaN and infinite values
        valid_pixels = background_pixels[np.isfinite(background_pixels)]
        
        if len(valid_pixels) == 0:
            print("Error: No valid background pixels found")
            return None
        
        # Calculate basic statistics
        stats = {
            'mean': np.mean(valid_pixels),
            'median': np.median(valid_pixels),
            'std': np.std(valid_pixels),
            'n_pixels': len(valid_pixels),
            'background_fraction': len(valid_pixels) / image_data.size
        }
        
        # Calculate resistant mean using your custom function
        resistant_result = resistant_mean(valid_pixels, threshold=5.0)
        stats['resistant_mean'] = resistant_result[0]
        stats['resistant_std'] = resistant_result[1]
        stats['resistant_std_of_mean'] = resistant_result[2]
        
        # Sigma-clipped mean (3-sigma, 3 iterations) - keep for comparison
        clipped_pixels = valid_pixels.copy()
        for _ in range(3):
            mean_val = np.mean(clipped_pixels)
            std_val = np.std(clipped_pixels)
            clip_mask = np.abs(clipped_pixels - mean_val) < 3 * std_val
            clipped_pixels = clipped_pixels[clip_mask]
            if len(clipped_pixels) == 0:
                break
        
        if len(clipped_pixels) > 0:
            stats['sigma_clipped_mean'] = np.mean(clipped_pixels)
            stats['sigma_clipped_std'] = np.std(clipped_pixels)
        else:
            stats['sigma_clipped_mean'] = stats['mean']
            stats['sigma_clipped_std'] = stats['std']
        
        return stats
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def find_image_pairs(original_dir, segmented_dir):
    """Find matching image pairs with specific naming pattern"""
    original_files = glob.glob(os.path.join(original_dir, "*.fits"))
    pairs = []
    
    print(f"Found {len(original_files)} original FITS files")
    
    for orig_file in original_files:
        basename = os.path.splitext(os.path.basename(orig_file))[0]
        
        # Try different segmentation file naming patterns
        segm_patterns = [
            f"{basename}_segmentation.fits",  # Main expected pattern
            f"{basename}_segm.fits",          # Alternative pattern
            f"{basename}.segm.fits",          # Another alternative
            f"segm_{basename}.fits"           # Yet another alternative
        ]
        
        segm_file_found = False
        for pattern in segm_patterns:
            segm_path = os.path.join(segmented_dir, pattern)
            if os.path.exists(segm_path):
                pairs.append((orig_file, segm_path))
                print(f"✓ Matched: {os.path.basename(orig_file)} -> {pattern}")
                segm_file_found = True
                break
        
        if not segm_file_found:
            print(f"✗ No match: {basename}")
            print(f"   Expected: {basename}_segmentation.fits")
    
    return pairs

def process_filter(filter_name, base_dir=".", verbose=True):
    """Process a single filter and return results"""
    original_dir = f"galaxies_train_{filter_name}_with_pedestal"
    segmented_dir = f"galaxies_train_{filter_name}_segmented_sextractor"
    
    # Adjust paths if base_dir is provided
    if base_dir != ".":
        original_dir = os.path.join(base_dir, original_dir)
        segmented_dir = os.path.join(base_dir, segmented_dir)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING FILTER: {filter_name}")
    print(f"{'='*60}")
    print(f"Original directory:  {original_dir}")
    print(f"Segmented directory: {segmented_dir}")
    print()
    
    # Check directories exist
    if not os.path.exists(original_dir):
        print(f"⚠ ERROR: Original directory not found: {original_dir}")
        return None
    
    if not os.path.exists(segmented_dir):
        print(f"⚠ ERROR: Segmented directory not found: {segmented_dir}")
        return None
    
    # Find image pairs
    pairs = find_image_pairs(original_dir, segmented_dir)
    
    if not pairs:
        print(f"\n⚠ No matching image pairs found for {filter_name}!")
        return None
    
    print(f"\n✅ Found {len(pairs)} matching image pairs for {filter_name}")
    print("=" * 60)
    
    # Process each pair
    all_results = []
    
    for i, (orig_file, segm_file) in enumerate(pairs, 1):
        basename = os.path.splitext(os.path.basename(orig_file))[0]
        
        if verbose:
            print(f"Processing {i}/{len(pairs)}: {basename}")
        
        stats = calculate_background_mean(orig_file, segm_file)
        
        if stats:
            stats['filename'] = basename
            stats['filter'] = filter_name
            all_results.append(stats)
            
            if verbose:
                print(f"  Regular mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
                print(f"  Resistant mean: {stats['resistant_mean']:.6f} ± {stats['resistant_std']:.6f}")
                print(f"  σ-clipped: {stats['sigma_clipped_mean']:.6f} ± {stats['sigma_clipped_std']:.6f}")
                print(f"  Background pixels: {stats['n_pixels']:,} ({stats['background_fraction']:.1%})")
        else:
            print(f"⚠ Failed to process: {basename}")
    
    return all_results

def main():
    """Main function - process all filters"""
    
    # ==========================================================================
    # CONFIGURATION - MODIFY THESE PATHS TO MATCH YOUR SETUP
    # ==========================================================================
    filters = ["VIS", "NISP_H", "NISP_J", "NISP_Y"]
    base_dir = "."  # Change this if your directories are in a different location
    output_file = "background_results_all_filters.csv"
    verbose = True
    
    # ==========================================================================
    
    print("=" * 80)
    print("MULTI-FILTER BACKGROUND CALCULATOR WITH RESISTANT MEAN")
    print("=" * 80)
    print(f"Filters to process: {', '.join(filters)}")
    print(f"Base directory: {base_dir}")
    print(f"Output file: {output_file}")
    print()
    
    all_filter_results = []
    filter_summaries = []
    
    # Process each filter
    for filter_name in filters:
        results = process_filter(filter_name, base_dir, verbose)
        
        if results:
            all_filter_results.extend(results)
            
            # Calculate summary statistics for this filter
            resistant_means = [r['resistant_mean'] for r in results if not np.isnan(r['resistant_mean'])]
            regular_means = [r['mean'] for r in results]
            sigma_clipped_means = [r['sigma_clipped_mean'] for r in results]
            
            if resistant_means:
                # Calculate overall resistant mean for this filter
                all_resistant_values = []
                for r in results:
                    if not np.isnan(r['resistant_mean']):
                        all_resistant_values.append(r['resistant_mean'])
                
                # Use resistant_mean function on the individual resistant means
                overall_resistant = resistant_mean(np.array(all_resistant_values), threshold=5.0)
                
                filter_summary = {
                    'filter': filter_name,
                    'n_images': len(results),
                    'mean_regular': np.mean(regular_means),
                    'std_regular': np.std(regular_means),
                    'mean_resistant': overall_resistant[0],
                    'std_resistant': overall_resistant[1],
                    'std_of_mean_resistant': overall_resistant[2],
                    'mean_sigma_clipped': np.mean(sigma_clipped_means),
                    'std_sigma_clipped': np.std(sigma_clipped_means),
                    'median_resistant': np.median(resistant_means),
                    'range_resistant_min': np.min(resistant_means),
                    'range_resistant_max': np.max(resistant_means)
                }
                filter_summaries.append(filter_summary)
                
                print(f"\n{'-'*60}")
                print(f"SUMMARY FOR {filter_name}")
                print(f"{'-'*60}")
                print(f"Number of images: {len(results)}")
                print(f"Mean background (regular):     {np.mean(regular_means):.6f} ± {np.std(regular_means):.6f}")
                print(f"Mean background (resistant):   {overall_resistant[0]:.6f} ± {overall_resistant[1]:.6f}")
                print(f"Std of resistant mean:         {overall_resistant[2]:.6f}")
                print(f"Mean background (σ-clipped):   {np.mean(sigma_clipped_means):.6f} ± {np.std(sigma_clipped_means):.6f}")
                print(f"Median resistant background:   {np.median(resistant_means):.6f}")
                print(f"Range (resistant):             {np.min(resistant_means):.6f} to {np.max(resistant_means):.6f}")
        else:
            print(f"\n⚠ No valid results for filter {filter_name}")
    
    # Overall summary across all filters
    if filter_summaries:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS ALL FILTERS")
        print(f"{'='*80}")
        
        all_resistant_filter_means = [fs['mean_resistant'] for fs in filter_summaries if not np.isnan(fs['mean_resistant'])]
        if all_resistant_filter_means:
            overall_resistant_all = resistant_mean(np.array(all_resistant_filter_means), threshold=5.0)
            print(f"Overall resistant mean across filters: {overall_resistant_all[0]:.6f} ± {overall_resistant_all[1]:.6f}")
            print(f"Std of overall resistant mean:          {overall_resistant_all[2]:.6f}")
        
        print("\nPer-filter resistant means:")
        for fs in filter_summaries:
            print(f"  {fs['filter']:8}: {fs['mean_resistant']:.6f} ± {fs['std_resistant']:.6f} (n={fs['n_images']})")
    
    # Save detailed results
    if all_filter_results and output_file:
        try:
            with open(output_file, 'w') as f:
                f.write("filter,filename,mean,std,median,resistant_mean,resistant_std,resistant_std_of_mean,")
                f.write("sigma_clipped_mean,sigma_clipped_std,n_pixels,bg_fraction\n")
                for result in all_filter_results:
                    f.write(f"{result['filter']},{result['filename']},{result['mean']:.6f},")
                    f.write(f"{result['std']:.6f},{result['median']:.6f},")
                    f.write(f"{result['resistant_mean']:.6f},{result['resistant_std']:.6f},")
                    f.write(f"{result['resistant_std_of_mean']:.6f},")
                    f.write(f"{result['sigma_clipped_mean']:.6f},{result['sigma_clipped_std']:.6f},")
                    f.write(f"{result['n_pixels']},{result['background_fraction']:.6f}\n")
            
            print(f"\n✅ Detailed results saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️ Warning: Could not save detailed results to {output_file}: {str(e)}")
    
    # Save filter summaries
    summary_file = "filter_summaries.csv"
    if filter_summaries:
        try:
            with open(summary_file, 'w') as f:
                f.write("filter,n_images,mean_regular,std_regular,mean_resistant,std_resistant,")
                f.write("std_of_mean_resistant,mean_sigma_clipped,std_sigma_clipped,")
                f.write("median_resistant,range_resistant_min,range_resistant_max\n")
                for fs in filter_summaries:
                    f.write(f"{fs['filter']},{fs['n_images']},{fs['mean_regular']:.6f},")
                    f.write(f"{fs['std_regular']:.6f},{fs['mean_resistant']:.6f},")
                    f.write(f"{fs['std_resistant']:.6f},{fs['std_of_mean_resistant']:.6f},")
                    f.write(f"{fs['mean_sigma_clipped']:.6f},{fs['std_sigma_clipped']:.6f},")
                    f.write(f"{fs['median_resistant']:.6f},{fs['range_resistant_min']:.6f},")
                    f.write(f"{fs['range_resistant_max']:.6f}\n")
            
            print(f"✅ Filter summaries saved to: {summary_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save filter summaries to {summary_file}: {str(e)}")
    
    return 0

if __name__ == "__main__":
    main()