import numpy as np
import os
import pdb
from astropy.io import fits
from astropy import wcs
import montage_wrapper as montage


def pix2coo(filename,xx,yy,ext):

    hdu = fits.open(filename)
    hdr = hdu[ext].header
    del hdr["CNPIX1"]
    del hdr["CNPIX2"]
    del hdr["PV1_5"]
    del hdr["PV1_6"]
    ww = wcs.WCS(hdr)
    
    centroid = []
    centroid.append([float("NaN"),float("NaN")]) #when you use a single element, you need to add an empty coordinate
    centroid.append([xx,yy])
    centroid = np.array(centroid, dtype=float)

    empty, coos = ww.wcs_pix2world(centroid,1)
    
    ra  = coos[0]
    dec = coos[1]
    return(ra,dec)  #[1] to remove the empty coordinate
    
    
def coo2pix(filename,ra,dec,ext):

    hdu = fits.open(filename)
    hdr = hdu[ext].header
    del hdr["CNPIX1"]
    del hdr["CNPIX2"]
    del hdr["PV1_5"]
    del hdr["PV1_6"]
    ww = wcs.WCS(hdr)
    
    coo = []
    coo.append([float("NaN"),float("NaN")]) #when you use a single element, you need to add an empty coordinate
    coo.append([ra,dec])
    coo = np.array(coo, dtype = float)
    
    empty, pixs = ww.wcs_world2pix(coo,1)

    pix_x = pixs[0]
    pix_y = pixs[1]
    return(pix_x,pix_y) #[1] to remove the empty coordinate
    

path_to_input = "./FITS-SingleFrames/" 
imageV1 = "EUC_LE1_VIS-65658-1-C_20230909T112702.000000Z_01_01_01.00.fh.MOBF.CRCLEAN.NLSCALE.fits.fz" #ext 3
imageV2 = "EUC_LE1_VIS-65658-1-C_20230909T112702.000000Z_01_01_01.00.fh.MOBF.CRCLEAN.NLSCALE.fits.fz" #ext 4
imageY  = "EUC_LE1_NISP-65658-1-C_20230909T112702.000000Z_01_04_01.00.fh.MODFN.fits.fz" #ext 13
imageJ  = "EUC_LE1_NISP-65658-1-C_20230909T112702.000000Z_01_02_01.00.fh.MODFN.fits.fz" #ext 13
imageH  = "EUC_LE1_NISP-65658-1-C_20230909T112702.000000Z_01_03_01.00.fh.MODFN.fits.fz" #ext 13
number_gals = 1000
side_pix_nir = 200
side_pix_opt = 600
path_input_single_ext = "./FITS-SingleFrames/single_extension_without_pedestal/" 
path_to_output = "./stamps/"


if os.path.exists(path_input_single_ext) == False:
    os.mkdir(path_input_single_ext)
    
if os.path.exists(path_to_output) == False:
    os.mkdir(path_to_output)
    
hdu_opt = fits.open(path_to_input+imageV1)
img_opt = hdu_opt[3].data
hdr_opt = hdu_opt[3].header
#fits.writeto(path_input_single_ext+"imageV1.fits",img_opt,hdr_opt,overwrite=True)
    
hdu_opt = fits.open(path_to_input+imageV2)
img_opt = hdu_opt[4].data
hdr_opt = hdu_opt[4].header
#fits.writeto(path_input_single_ext+"imageV2.fits",img_opt,hdr_opt,overwrite=True)
    
hdu_nir = fits.open(path_to_input+imageY)
img_nir = hdu_nir[13].data
hdr_nir = hdu_nir[13].header
#fits.writeto(path_input_single_ext+"imageY.fits",img_nir,hdr_nir,overwrite=True)

hdu_nir = fits.open(path_to_input+imageJ)
img_nir = hdu_nir[13].data
hdr_nir = hdu_nir[13].header
#fits.writeto(path_input_single_ext+"imageJ.fits",img_nir,hdr_nir,overwrite=True)

hdu_nir = fits.open(path_to_input+imageH)
img_nir = hdu_nir[13].data
hdr_nir = hdu_nir[13].header
#fits.writeto(path_input_single_ext+"imageH.fits",img_nir,hdr_nir,overwrite=True)


cont_gal = 0
while cont_gal < number_gals:
    
    rnd_img = np.random.randint(0,2)
    if rnd_img == 0:
        opt_img = "imageV1.fits"
    else:
        opt_img = "imageV2.fits"
        
    hdu_opt = fits.open(path_input_single_ext+opt_img)
    img_opt = hdu_opt[0].data
    hdr_opt = hdu_opt[0].header
    dim1_opt = hdr_opt["NAXIS1"] #this is the x-axis, but it is the second dimension in img_opt
    dim2_opt = hdr_opt["NAXIS2"] #this is the y-axis, but it is the first  dimension in img_opt
    
    rnd_opt_y = np.random.randint(0, dim1_opt)
    rnd_opt_x = np.random.randint(0, dim2_opt)
    
    if  rnd_opt_x > int(np.round(side_pix_opt/2)) and rnd_opt_x < dim2_opt-int(np.round(side_pix_opt/2)) and \
        rnd_opt_y > int(np.round(side_pix_opt/2)) and rnd_opt_y < dim1_opt-int(np.round(side_pix_opt/2)):
        
        ra, dec = pix2coo(path_input_single_ext+opt_img, rnd_opt_x, rnd_opt_y, 0)
        rnd_nir_x, rnd_nir_y = coo2pix(path_input_single_ext+"imageH.fits", ra, dec, 0)
        dim1_nir = hdr_nir["NAXIS1"]
        dim2_nir = hdr_nir["NAXIS2"]
        
        if  rnd_nir_x > int(np.round(side_pix_nir/2)) and rnd_nir_x < dim2_nir-int(np.round(side_pix_nir/2)) and \
            rnd_nir_y > int(np.round(side_pix_nir/2)) and rnd_nir_y < dim1_nir-int(np.round(side_pix_nir/2)):

            montage.mSubimage_pix(path_input_single_ext+opt_img, path_to_output+str(cont_gal)+"_I.fits", rnd_opt_x-int(np.round(side_pix_opt/2)), rnd_opt_y-int(np.round(side_pix_opt/2)), side_pix_opt, hdu=0)
            
            hdu = fits.open(path_to_output+str(cont_gal)+"_I.fits")
            img_flip = np.flip(hdu[0].data, axis = 1) #because otherwise the images only coincide when doing ds9 Lock coordinates
            hdr = hdu[0].header
            fits.writeto(path_to_output+str(cont_gal)+"_I.fits", img_flip, header=hdr, overwrite=True)
            hdu.close()

            montage.mSubimage_pix(path_input_single_ext+"imageY.fits", path_to_output+str(cont_gal)+"_Y.fits", rnd_nir_x-int(np.round(side_pix_nir/2)), rnd_nir_y-int(np.round(side_pix_nir/2)), side_pix_nir, hdu=0)
            montage.mSubimage_pix(path_input_single_ext+"imageJ.fits", path_to_output+str(cont_gal)+"_J.fits", rnd_nir_x-int(np.round(side_pix_nir/2)), rnd_nir_y-int(np.round(side_pix_nir/2)), side_pix_nir, hdu=0)
            montage.mSubimage_pix(path_input_single_ext+"imageH.fits", path_to_output+str(cont_gal)+"_H.fits", rnd_nir_x-int(np.round(side_pix_nir/2)), rnd_nir_y-int(np.round(side_pix_nir/2)), side_pix_nir, hdu=0)

            cont_gal += 1
        else:
            print(f"Failed at NIR condition: rnd_nir_x={rnd_nir_x}, rnd_nir_y={rnd_nir_y}, side_pix_nir={side_pix_nir}")
    else:
        print(f"Failed at OPT condition: rnd_opt_x={rnd_opt_x}, rnd_opt_y={rnd_opt_y}, side_pix_opt={side_pix_opt}")
