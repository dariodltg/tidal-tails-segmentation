import glob
import numpy as np
import montage_wrapper as montage
import os.path
from astropy.io import fits



size_VIS = 600
size_NISP = 200
redshift_variation_numbers= 10
background_images_path = "backgrounds/"

input_VIS_folder = "make_mock_tidal_streams_VIS/"
output_VIS_folder= "../segmentation_training/galaxies_train_VIS/"
background_VIS_image = "IC342_VIS.fits"

input_NISP_folders = ["make_mock_tidal_streams_NISP_H/", "make_mock_tidal_streams_NISP_J/", "make_mock_tidal_streams_NISP_Y/"]
output_NISP_folders = ["../segmentation_training/galaxies_train_NISP_H/","../segmentation_training/galaxies_train_NISP_J/","../segmentation_training/galaxies_train_NISP_Y/"]
background_NISP_images = ["IC342_NISP_H.fits","IC342_NISP_J.fits","IC342_NISP_Y.fits"]


for output_folder in output_NISP_folders:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

if not os.path.exists(output_VIS_folder):
    os.mkdir(output_VIS_folder)


# Open VIS background image
hdu_background_VIS = fits.open(background_images_path+background_VIS_image)
img_background_VIS = hdu_background_VIS[0].data
hdr_backgground_VIS = hdu_background_VIS[0].header
dim1_background_VIS = hdr_backgground_VIS["NAXIS1"]
dim2_background_VIS = hdr_backgground_VIS["NAXIS2"]

#Open NISP images
hdu_background_NISP_H = fits.open(background_images_path+background_NISP_images[0])
img_background_NISP_H = hdu_background_NISP_H[0].data
hdu_background_NISP_J = fits.open(background_images_path+background_NISP_images[1])
img_background_NISP_J = hdu_background_NISP_J[0].data
hdu_background_NISP_Y = fits.open(background_images_path+background_NISP_images[2])
img_background_NISP_Y = hdu_background_NISP_Y[0].data

def split_list(list, sublist_size):
    return [list[i:i + sublist_size] for i in range(0, len(list), sublist_size)]


def insert_tidal_tails():
    
    input_files_vis = glob.glob(input_VIS_folder+"galaxy_and_stream_convolved_*.fits")
    input_files_vis_splitted_by_redshift = split_list(input_files_vis, redshift_variation_numbers)

    input_files_nisp_h = glob.glob(input_NISP_folders[0]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_h_splitted_by_redshift = split_list(input_files_nisp_h, redshift_variation_numbers)

    input_files_nisp_j = glob.glob(input_NISP_folders[1]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_j_splitted_by_redshift = split_list(input_files_nisp_j, redshift_variation_numbers)

    input_files_nisp_y = glob.glob(input_NISP_folders[2]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_y_splitted_by_redshift = split_list(input_files_nisp_y, redshift_variation_numbers)

    
    for input_file_vis_sublist,input_file_nisp_h_sublist,input_file_nisp_j_sublist, input_file_nisp_y_sublist in zip(input_files_vis_splitted_by_redshift, input_files_nisp_h_splitted_by_redshift, input_files_nisp_j_splitted_by_redshift, input_files_nisp_y_splitted_by_redshift):
        # Getting the random center of the background image to cutout
        flag_cutted = False
        while flag_cutted == False:
            #I obtain the new random centroid within the background image
            x_center = np.random.randint(0, dim1_background_VIS)
            y_center = np.random.randint(0, dim2_background_VIS)

            #I will take a cutout in the new centroid position with the dimensions of the simulated galaxy
            pix_in_x_halfsize = int(size_VIS/2)
            pix_in_y_halfsize = int(size_VIS/2)
            cutout = img_background_VIS[y_center-pix_in_y_halfsize:y_center+pix_in_y_halfsize, x_center-pix_in_x_halfsize:x_center+pix_in_x_halfsize]

            #I will accept this new image if less than 10% of the pixels are zeros and the dimensions are all right
            if np.count_nonzero(cutout == 0) >= (size_VIS*size_VIS)/10 or cutout.shape[0] != size_VIS or cutout.shape[1] != size_VIS or np.isnan(cutout).any():
                flag_cutted = False
            else:
                flag_cutted = True

        # First the VIS filter
        for input_file_vis in input_file_vis_sublist:
            hdu_sim = fits.open(input_file_vis)
            img_sim = hdu_sim[1].data
            print("Inserting background image to: " + input_file_vis)
            #I create the cutout itself
            new_filename = output_folder+input_file_vis[len(input_VIS_folder):-5]+"_in_"+background_VIS_image[:-5]+".fits"
            montage.mSubimage_pix(background_images_path+background_VIS_image,new_filename, x_center-pix_in_x_halfsize, y_center-pix_in_y_halfsize, hdu = 0, xpixsize = (pix_in_x_halfsize*2)-1, ypixsize = (pix_in_y_halfsize*2)-1 )
            hdu_cutout = fits.open(new_filename)
            img_cutout = hdu_cutout[0].data
            hdr_cutout = hdu_cutout[0].header
            fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)

        # Second, the NISP filters
        x_center = int(x_center/3)
        y_center = int(y_center/3)
        pix_in_x_halfsize = int(size_NISP/2)
        pix_in_y_halfsize = int(size_NISP/2)
        
        for input_file_nisp_h in input_file_nisp_h_sublist:
            hdu_sim = fits.open(input_file_nisp_h)
            img_sim = hdu_sim[1].data
            print("Inserting background image to: " + input_file_nisp_h)
            #I create the cutout itself
            new_filename = output_folder+input_file_nisp_h[len(input_NISP_folders[0]):-5]+"_in_"+background_NISP_images[0][:-5]+".fits"
            montage.mSubimage_pix(background_images_path+background_NISP_images[0],new_filename, x_center-pix_in_x_halfsize, y_center-pix_in_y_halfsize, hdu = 0, xpixsize = (pix_in_x_halfsize*2)-1, ypixsize = (pix_in_y_halfsize*2)-1 )
            hdu_cutout = fits.open(new_filename)
            img_cutout = hdu_cutout[0].data
            hdr_cutout = hdu_cutout[0].header
            fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)

        for input_file_nisp_j in input_file_nisp_j_sublist:
            hdu_sim = fits.open(input_file_nisp_j)
            img_sim = hdu_sim[1].data
            print("Inserting background image to: " + input_file_nisp_j)
            #I create the cutout itself
            new_filename = output_folder+input_file_nisp_j[len(input_NISP_folders[1]):-5]+"_in_"+background_NISP_images[1][:-5]+".fits"
            montage.mSubimage_pix(background_images_path+background_NISP_images[1],new_filename, x_center-pix_in_x_halfsize, y_center-pix_in_y_halfsize, hdu = 0, xpixsize = (pix_in_x_halfsize*2)-1, ypixsize = (pix_in_y_halfsize*2)-1 )
            hdu_cutout = fits.open(new_filename)
            img_cutout = hdu_cutout[0].data
            hdr_cutout = hdu_cutout[0].header
            fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)

        for input_file_nisp_y in input_file_nisp_y_sublist:
            hdu_sim = fits.open(input_file_nisp_y)
            img_sim = hdu_sim[1].data
            print("Inserting background image to: " + input_file_nisp_y)
            #I create the cutout itself
            new_filename = output_folder+input_file_nisp_y[len(input_NISP_folders[2]):-5]+"_in_"+background_NISP_images[2][:-5]+".fits"
            montage.mSubimage_pix(background_images_path+background_NISP_images[2],new_filename, x_center-pix_in_x_halfsize, y_center-pix_in_y_halfsize, hdu = 0, xpixsize = (pix_in_x_halfsize*2)-1, ypixsize = (pix_in_y_halfsize*2)-1 )
            hdu_cutout = fits.open(new_filename)
            img_cutout = hdu_cutout[0].data
            hdr_cutout = hdu_cutout[0].header
            fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)