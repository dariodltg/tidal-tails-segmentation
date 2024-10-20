from collections import defaultdict
import glob
import numpy as np
import montage_wrapper as montage
import os.path
from astropy.io import fits


size_VIS = 600
size_NISP = 200

input_VIS_folder = "make_mock_tidal_streams_VIS/"
output_VIS_folder= "../segmentation_training/v5/galaxies_train_VIS/"

input_NISP_folders = ["make_mock_tidal_streams_NISP_H/", "make_mock_tidal_streams_NISP_J/", "make_mock_tidal_streams_NISP_Y/"]
output_NISP_folders = ["../segmentation_training/v5/galaxies_train_NISP_H/","../segmentation_training/v5/galaxies_train_NISP_J/","../segmentation_training/v5/galaxies_train_NISP_Y/"]

background_stamps_folder = "stamps/"

for output_folder in output_NISP_folders:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

if not os.path.exists(output_VIS_folder):
    os.mkdir(output_VIS_folder)


def split_list(file_list):
    grouped_files = defaultdict(list)

    # Procesar cada archivo y agruparlos
    for file in file_list:
        # Extraer el número que sigue a "galaxy_and_stream_convolved_"
        parts = file.split('/')[1].split('_')
        number = parts[4]  # El número está en la posición 4 después de dividir por '_'
        
        # Añadir el archivo a la lista correspondiente en el diccionario
        grouped_files[number].append(file)

    # Convertir el diccionario a una lista de listas
    grouped_files_list = list(grouped_files.values())
    return grouped_files_list


def insert_tidal_tails():
    
    input_files_vis = glob.glob(input_VIS_folder+"galaxy_and_stream_convolved_*.fits")
    input_files_vis.sort()
    input_files_vis_splitted_by_redshift = split_list(input_files_vis)

    input_files_nisp_h = glob.glob(input_NISP_folders[0]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_h.sort()
    input_files_nisp_h_splitted_by_redshift = split_list(input_files_nisp_h)

    input_files_nisp_j = glob.glob(input_NISP_folders[1]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_j.sort()
    input_files_nisp_j_splitted_by_redshift = split_list(input_files_nisp_j)

    input_files_nisp_y = glob.glob(input_NISP_folders[2]+"galaxy_and_stream_convolved_*.fits")
    input_files_nisp_y.sort()
    input_files_nisp_y_splitted_by_redshift = split_list(input_files_nisp_y)

    VIS_CENTER = int(size_VIS/2)
    NISP_CENTER = int(size_NISP/2)
    pix_in_x_halfsize = int(size_VIS/2)
    pix_in_y_halfsize = int(size_VIS/2)
    index = 0
    for input_file_vis_sublist,input_file_nisp_h_sublist,input_file_nisp_j_sublist, input_file_nisp_y_sublist in zip(input_files_vis_splitted_by_redshift, input_files_nisp_h_splitted_by_redshift, input_files_nisp_j_splitted_by_redshift, input_files_nisp_y_splitted_by_redshift):

        vis_background_image = background_stamps_folder+str(index) +"_I.fits"
        nisp_y_background_image = background_stamps_folder+str(index) +"_Y.fits"
        nisp_j_background_image = background_stamps_folder+str(index) +"_J.fits"
        nisp_h_background_image = background_stamps_folder+str(index) +"_H.fits"

        # First the VIS filter
        for input_file_vis in input_file_vis_sublist:
            try:
                hdu_sim = fits.open(input_file_vis)
                img_sim = hdu_sim[1].data
                print("Inserting background image to: " + input_file_vis)
                #I create the cutout itself
                new_filename = output_VIS_folder+input_file_vis[len(input_VIS_folder):-5]+"_in_background"+str(index)+".fits"
                montage.mSubimage_pix(vis_background_image, new_filename, 0, 0, hdu = 0, xpixsize = size_VIS, ypixsize = size_VIS )            
                hdu_cutout = fits.open(new_filename)
                img_cutout = hdu_cutout[0].data
                hdr_cutout = hdu_cutout[0].header
                fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)
            except Exception as ex:
                print(ex)
        
        for input_file_nisp_h in input_file_nisp_h_sublist:
            try:
                hdu_sim = fits.open(input_file_nisp_h)
                img_sim = hdu_sim[1].data
                print("Inserting background image to: " + input_file_nisp_h)
                #I create the cutout itself
                new_filename = output_NISP_folders[0]+input_file_nisp_h[len(input_NISP_folders[0]):-5]+"_in_background"+str(index)+".fits"
                montage.mSubimage_pix(nisp_h_background_image,new_filename, 0, 0, hdu = 0, xpixsize = size_NISP, ypixsize = size_NISP )
                hdu_cutout = fits.open(new_filename)
                img_cutout = hdu_cutout[0].data
                hdr_cutout = hdu_cutout[0].header
                fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)
            except Exception as ex:
                print(ex)

        for input_file_nisp_j in input_file_nisp_j_sublist:
            try:
                hdu_sim = fits.open(input_file_nisp_j)
                img_sim = hdu_sim[1].data
                print("Inserting background image to: " + input_file_nisp_j)
                #I create the cutout itself
                new_filename = output_NISP_folders[1]+input_file_nisp_j[len(input_NISP_folders[1]):-5]+"_in_background"+str(index)+".fits"
                montage.mSubimage_pix(nisp_j_background_image,new_filename, 0, 0, hdu = 0, xpixsize = size_NISP, ypixsize = size_NISP )
                hdu_cutout = fits.open(new_filename)
                img_cutout = hdu_cutout[0].data
                hdr_cutout = hdu_cutout[0].header
                fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)
            except Exception as ex:
                print(ex)

        for input_file_nisp_y in input_file_nisp_y_sublist:
            try:
                hdu_sim = fits.open(input_file_nisp_y)
                img_sim = hdu_sim[1].data
                print("Inserting background image to: " + input_file_nisp_y)
                #I create the cutout itself
                new_filename = output_NISP_folders[2]+input_file_nisp_y[len(input_NISP_folders[2]):-5]+"_in_background"+str(index)+".fits"
                montage.mSubimage_pix(nisp_y_background_image,new_filename, 0, 0, hdu = 0, xpixsize = size_NISP, ypixsize = size_NISP )
                hdu_cutout = fits.open(new_filename)
                img_cutout = hdu_cutout[0].data
                hdr_cutout = hdu_cutout[0].header
                fits.writeto(new_filename,img_cutout+img_sim,hdr_cutout,overwrite=True)
            except Exception as ex:
                print(ex)
            
        index = index+1
    
        