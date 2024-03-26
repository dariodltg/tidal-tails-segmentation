#!/bin/bash
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License (LGPL)
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License with this program. If not, see <http://www.gnu.org/licenses/>.





# Force the script to crash in the case of an error in any of the
# commands.
set -e


bdir_VIS="make_mock_tidal_streams_VIS"
bdir_NISP="make_mock_tidal_streams_NISP"
bdir_check_cat="catalogs_for_checking"
psf_VIS="psfs_connor/star_VIS_connor.fits" #../dario_2024/dario_IC342_VIS_star.fits"
psf_NISP_Y="psfs_connor/star_NISP_Y_connor.fits" #"../dario_2024/dario_IC342_NISP_Y_star.fits"
psf_NISP_J="psfs_connor/star_NISP_J_connor.fits" #"../dario_2024/dario_IC342_NISP_J_star.fits"
psf_NISP_H="psfs_connor/star_NISP_H_connor.fits" #"../dario_2024/dario_IC342_NISP_H_star.fits"
flag_gaussian_noise="False" #create or not images adding gaussian noise
output_size_VIS=600 #[pix in VIS]
output_size_NISP=200 #[pix in NISP]
x_cen_VIS=301 #[pix in VIS]
y_cen_VIS=301 #[pix in VIS]
x_cen_NISP=101 #[pix in NISP]
y_cen_NISP=101 #[pix in NISP]
zeropoint_VIS=30.132
zeropoint_NISP=30.0
pix_scale_VIS=0.1
pix_scale_NISP=0.3
bckg_mag_VIS=29.5
bckg_mag_NISP=28.5
bands="VIS NISP_Y NISP_J NISP_H"
redshifts=(0.05 0.1 0.15 0.2 0.25 0.4 0.6 0.8 1) #range of possible redshifts
phys_scales=(0.977 1.844 2.614 3.300 3.910 5.373 6.685 7.508 8.008) #range of physical scales at each redshift
comparing_to_05=(1 0.530 0.347 0.296 0.250 0.182 0.146 0.130 0.122) #comparing physical scales to the one at z = 0.05
number_galaxies=2 #number of systems (galaxy + tidal stream) to be created

#TIDAL STREAM LIMITS
mag_stream_bright=19.8
mag_stream_faint=20.2
min_rr_stream=15 #[pix in VIS]
max_rr_stream=120 #[pix in VIS]
min_width_stream=15 #[pix in VIS], check also min_rr_stream 
max_width_stream=100 #[pix in VIS] 
min_azimuthal_width=45 #degrees
max_azimuthal_width=270 #degrees
sigma_over_noise=1.2

#BULGE LIMITS
mag_bulge_bright_ini=19
mag_bulge_faint_ini=21
re_bulge_min_ini=1 #[kpc]
re_bulge_max_ini=6 #[kpc]

#DISK_LIMITS
mag_disk_bright_ini=17
mag_disk_faint_ini=19
re_disk_min_ini=2 #[kpc]
re_disk_max_ini=6 #[kpc]


# Make the build directory (if it doens't exist).
if ! [ -d "$bdir_VIS" ]; then mkdir "$bdir_VIS"; fi
if ! [ -d "$bdir_NISP" ]; then mkdir "$bdir_NISP"; fi
if ! [ -d "$bdir_check_cat" ]; then mkdir "$bdir_check_cat"; fi


for ii in `seq $number_galaxies`
do  
    for kk in {0..8}
    do
    
        zz=${redshifts[kk]}
        phys_scale=${phys_scales[kk]}
        
        for band in $bands
        do  
        
            case $band in
                "VIS")
                pix_scale=$pix_scale_VIS
                zeropoint=$zeropoint_VIS
                bckg_mag=$bckg_mag_VIS
                psf=$psf_VIS
                bdir=$bdir_VIS
                output_size=$output_size_VIS
                x_cen=$x_cen_VIS
                y_cen=$y_cen_VIS
                ;;
                "NISP_Y")
                pix_scale=$pix_scale_NISP
                zeropoint=$zeropoint_NISP
                bckg_mag=$bckg_mag_NISP
                psf=$psf_NISP_Y
                bdir=$bdir_NISP
                output_size=$output_size_NISP
                x_cen=$x_cen_NISP
                y_cen=$y_cen_NISP
                ;;
                "NISP_J")
                pix_scale=$pix_scale_NISP
                zeropoint=$zeropoint_NISP
                bckg_mag=$bckg_mag_NISP
                bdir=$bdir_NISP
                psf=$psf_NISP_J
                output_size=$output_size_NISP
                x_cen=$x_cen_NISP
                y_cen=$y_cen_NISP
                ;;
                "NISP_H")
                pix_scale=$pix_scale_NISP
                zeropoint=$zeropoint_NISP
                bckg_mag=$bckg_mag_NISP
                bdir=$bdir_NISP
                psf=$psf_NISP_H
                output_size=$output_size_NISP
                x_cen=$x_cen_NISP
                y_cen=$y_cen_NISP
            esac
            
            kpc_to_pix=$(astarithmetic -q 1 $pix_scale / set-aa 1 $phys_scale / set-bb aa bb x)
            pix_to_kpc=$(astarithmetic -q 1 $kpc_to_pix /)
            
            if [ $kk -eq 0 ] && [ $band == "VIS" ]; then
            
                #GALAXY BULGE=============
                #=to obtain a 2-decimal number
                mag_bulge_bright=$(astarithmetic -q $mag_bulge_bright_ini int32 100 x)
                mag_bulge_faint=$(astarithmetic -q $mag_bulge_faint_ini int32 100 x)
                mag_bulge=`shuf -i $mag_bulge_bright-$mag_bulge_faint -n 1` 
                mag_bulge=$(astarithmetic -q $mag_bulge float32 100 /)
                #=to obtain a 2-decimal number
                re_bulge_min=$(astarithmetic -q $re_bulge_min_ini int32 100 x)
                re_bulge_max=$(astarithmetic -q $re_bulge_max_ini int32 100 x)
                re_bulge=`shuf -i $re_bulge_min-$re_bulge_max -n 1` 
                re_bulge=$(astarithmetic -q $re_bulge float32 100 /)
                
                #==
                pa_bulge=0 
                ar_bulge=1
                #=========================
                
                #GALAXY DISK=============
                #=to obtain a 2-decimal number
                mag_disk_bright=$(astarithmetic -q $mag_disk_bright_ini int32 100 x)
                mag_disk_faint=$(astarithmetic -q $mag_disk_faint_ini int32 100 x)
                mag_disk=`shuf -i $mag_disk_bright-$mag_disk_faint -n 1` 
                mag_disk=$(astarithmetic -q $mag_disk float32 100 /)
                #=to obtain a 2-decimal number
                re_disk_min=$(astarithmetic -q $re_disk_min_ini int32 100 x)
                re_disk_max=$(astarithmetic -q $re_disk_max_ini int32 100 x)
                re_disk=`shuf -i $re_disk_min-$re_disk_max -n 1` 
                re_disk=$(astarithmetic -q $re_disk float32 100 /)
                
                #==
                pa_disk=`shuf -i 0-359 -n 1` #obtaining one number between 0 and 359
                ar_disk=`shuf -i 1-90  -n 1` #obtaining one number between 1 (to avoid numerical errors) and 90
                ar_disk=$(echo "scale=2; ($ar_disk/100)" | bc) #dividing the previously obtained number by 100
                ar_disk=$(astarithmetic -q $ar_disk 0.1 +) #adding a minimum axis ratio of 0.1 in order not to be "too" edge-on
                #=========================
                
                #TIDAL STREAM==============
                
                #randomizing the stream's magnitude
                tmp=`shuf -i 1-1000 -n 1` #obtaining one number between 1 (to avoid numerical errors for the axis ratio) and 1000
                offset_from_bright_mag1=$(echo "scale=3; ($tmp/1000)" | bc) #dividing the previously obtained number by 1000
                offset_from_bright_mag2=$(astarithmetic -q $mag_stream_faint $mag_stream_bright - $offset_from_bright_mag1 x)
                mag_stream=$(astarithmetic -q $mag_stream_bright $offset_from_bright_mag2 +)
                
                #as the position of tidal tails is specified in the initial conditions in units of VIS pixels
                ratio_pixels=$(astarithmetic -q $pix_scale $pix_scale_VIS /)
                
	            #randomizing the initial radial position of the stream
	            rr_stream=`shuf -i $min_rr_stream-$max_rr_stream -n 1`
	            rr_stream_kpc=$(astarithmetic -q $rr_stream $pix_to_kpc x)
	       
	            #randomizing the width of the tidal stream
	            width_stream=`shuf -i $min_width_stream-$max_width_stream -n 1`
	            width_stream_kpc=$(astarithmetic -q $width_stream $pix_to_kpc x)
                
                #obtaining random axis ratio for the stream
                tmp=`shuf -i 50-1000 -n 1` #obtaining one number between 1 (to avoid numerical errors for the axis ratio) and 1000
                ar_stream=$(echo "scale=3; ($tmp/1000)" | bc) #dividing the previously obtained number by 1000
                
                #obtaining random position angle for the stream
                pa_stream=`shuf -i 0-359 -n 1` #obtaining one number between 1 (to avoid numerical errors for the axis ratio) and 1000
                
                #randomazing the stream's position angle according to the conditions I impose
                flag_to_get_min_azimuthal_width=0
                while [ $flag_to_get_min_azimuthal_width -eq 0 ]; do
                    azimuthal_width=`shuf -i $min_azimuthal_width-$max_azimuthal_width -n 1` #obtaining one number between min_azimuthal_width and max_azimuthal_width
                    potential_ini_angle=`shuf -i 0-359 -n 1` #obtaining one number between 0 and 359
                    potential_end_angle=$(astarithmetic -q $potential_ini_angle float32 $azimuthal_width float32 +) #without telling that these are floating point numbers, it's not working well
                    potential_end_angle_int=$(astarithmetic -q $potential_end_angle int32)
                    
                    #making sure that the stream will have the minimum azimuthal width, otherwise I repeat the while loop
                    ini_angle_to_be=$(astarithmetic -q $potential_end_angle_int 360 -)
                    dist_to_be=$(astarithmetic -q $potential_ini_angle $ini_angle_to_be -)
                    if [ $dist_to_be -ge $min_azimuthal_width ]; then
                        flag_to_get_min_azimuthal_width=`expr $flag_min_azimuthal_width + 1`
                    fi
                done

                #distinguishing between theh possible scenarios
                if [ $potential_end_angle_int -ge 360 ]; then
                    ini_angle=$(astarithmetic -q $potential_end_angle_int 360 -)
                    end_angle=$potential_ini_angle
                else       
                    ini_angle=$potential_ini_angle
                    end_angle=$potential_end_angle_int
                fi
                
                #===========================
                
                #starting the output catalog
                cat_output="$bdir_check_cat"/stream_characteristics_"$ii".txt
            	echo "#galaxy_number z band mag_bulge re_bulge re_bulge_pix ar_bulge pa_bulge mag_disk re_disk re_disk_pix ar_disk pa_disk mag_stream rr_stream width_stream azimuthal_width ar pa ini_angle end_angle width_feature" > "$cat_output"
            	
            fi
             
            #creating the galaxy=======================================================================================================
            re_bulge_pix=$(astarithmetic -q $re_bulge $kpc_to_pix x)
            re_disk_pix=$(astarithmetic -q $re_disk $kpc_to_pix x)
            
            cat="$bdir"/cat_"$ii"_"$zz"_"$band".txt
            echo "0 $x_cen $y_cen 1 $re_bulge_pix         4 $pa_bulge         $ar_bulge        $mag_bulge    5" >   "$cat"     # Bulge
            echo "1 $x_cen $y_cen 1 $re_disk_pix          1 $pa_disk          $ar_disk         $mag_disk     5" >>  "$cat"     # Disk
                   
            galaxy=galaxy_"$ii"_"$zz"_"$band".fits
            echo astmkprof "$cat" --mergedsize=$output_size,$output_size --oversample=1 --zeropoint=$zeropoint --type="float32" --output="$bdir"/"$galaxy"
            astmkprof "$cat" --mergedsize=$output_size,$output_size --oversample=1 --zeropoint=$zeropoint --type="float32" --output="$bdir"/"$galaxy"
            #============================================================================================================================
            
            #creating the tidal tail=====================================================================================================
	        rr_stream_pix=$(astarithmetic -q $rr_stream_kpc $kpc_to_pix x)
	        width_stream_pix=$(astarithmetic -q $width_stream_kpc $kpc_to_pix x)
	        
            #It is important that the radial and azimuthal images have the same radius ($rr_stream_pix) and end_radius ($to_where)
            to_where=$(astarithmetic -q $rr_stream_pix $width_stream_pix +)
            midpoint_of_stream=$(astarithmetic -q $to_where $rr_stream_pix - 2 / $rr_stream_pix +)
            #============================================================================================================================
	        
            #STREAM CONSTANT BRIGHTNESS============================================================================================================
            #the integrated magnitude of the sum of all pixels belonging to the stream will be $mag_stream
            cat="$bdir"/cat_stream_"$ii"_"$zz"_"$band".txt
            echo "0 $x_cen $y_cen 6 $rr_stream_pix 0 $pa_stream $ar_stream $mag_stream $to_where" > "$cat" # Stream constant brightness

            #--turnitinp makes that the end radius is not in units of the input radius but in pixels
            #--circumwidth gives you the width of your profile, while $to_where gives me the starting point
            stream_const=stream_const_profile_"$ii"_"$zz"_"$band".fits
            astmkprof "$cat" --mergedsize=$output_size,$output_size --oversample=1 --tunitinp --zeropoint=$zeropoint --circumwidth=$width_stream_pix --type="float32" --output="$bdir"/"$stream_const"
            #============================================================================================================================
            
            #RADIAL DISTANCE IMAGE============================================================================================================
            cat="$bdir"/cat_stream1_"$ii"_"$zz"_"$band".txt
            echo "1 $x_cen $y_cen 7 $rr_stream_pix 0 $pa_stream $ar_stream $mag_stream $to_where" > "$cat" # Radial dist

            radial_profile=radial_profile_"$ii"_"$zz"_"$band".fits
            astmkprof "$cat" --mergedsize=$output_size,$output_size --oversample=1 --tunitinp --zeropoint=$zeropoint --type="float32" --output="$bdir"/"$radial_profile"
            #============================================================================================================================
            
            #AZIMUTHAL DISTANCE IMAGE============================================================================================================
            cat="$bdir"/cat_stream2_"$ii"_"$zz"_"$band".txt
            echo "2 $x_cen $y_cen 9 $rr_stream_pix 0 $pa_stream $ar_stream $mag_stream $to_where" > "$cat" # Azimuthal dist 

            azimuthal_profile=azimuthal_profile_"$ii"_"$zz"_"$band".fits
            astmkprof "$cat" --mergedsize=$output_size,$output_size --oversample=1 --tunitinp --zeropoint=$zeropoint --type="float32" --output="$bdir"/"$azimuthal_profile"
            #============================================================================================================================
            
            #I take a radial distance from $rr_stream_pix to $to_where (i.e. coincident where the stream is)
            #and I do the absolute value to its middle point
            #and then I invert it to obtain a biggest value in the central part of the stream
            #Thus this is an image of the elliptical distance with some additions for to have a larger value in the stream's center
            #I take an azimuthal distance from $ini_angle to $end_angle
            #I set all the pixels that are not the stream itself to zero in order to 
            #have a label image and also to sum this image up to the galaxy's image
            output_tidal_image1=tidal_image_"$ii".fits
            astarithmetic "$bdir"/"$radial_profile" -h1 set-r \
            "$bdir"/"$azimuthal_profile" -h1 set-az \
            r r $rr_stream_pix lt r $to_where gt or nan where $midpoint_of_stream - abs set-rrange \
            1 rrange / set-ring ring az $ini_angle lt az $end_angle gt or nan where \
            set-tidal tidal tidal isblank 0 where --output="$bdir"/"$output_tidal_image1"
            
            #I add a random offset in order not to always modulated the same in the same spatial regions
            if [ $kk -eq 0 ] && [ $band == "VIS" ]; then #for adding the same modulation to all images
                azimuthal_offset=`shuf -i 0-359 -n 1` #obtaining one number between 0 and 359
            fi
            output_azimuthal_with_offset=azimuthal_profile_"$ii"_with_offset.fits
            astarithmetic "$bdir"/"$azimuthal_profile" -h1 $azimuthal_offset + --output="$bdir"/"$output_azimuthal_with_offset"
            #modulated the azimuthal image by a sin(2x) function, and using the absolute value to avoid negative values    
            output_azimuthal_modulated=azimuthal_profile_"$ii"_"$zz"_"$band"_modulated.fits
            astarithmetic "$bdir"/"$output_azimuthal_with_offset" -h1 set-az \
            az 2 x sin abs --output="$bdir"/"$output_azimuthal_modulated"
            #applying the modulation to the tidal feature
            output_tidal_modulated=tidal_image_modulated_"$ii".fits
            astarithmetic "$bdir"/"$output_azimuthal_modulated" -h1 set-az_mod \
            "$bdir"/"$output_tidal_image1" -h1 set-tidal \
            az_mod tidal x --output="$bdir"/"$output_tidal_modulated"
            
            #I normalize to the very big pixels by the value which is 10 times bigger than median value of pixels that appear *only* in the tidal tail
            output_tidal_image_normalized=tidal_image_normalized_"$ii".fits
            astarithmetic "$bdir"/"$output_tidal_modulated" -h1 set-img img img 0 eq nan where --output="$bdir"/tmp.fits
            median_output_tidal_image1=$(aststatistics "$bdir"/tmp.fits --median)
            threshold_for_bright_pix=$(astarithmetic -q $median_output_tidal_image1 10 x)
            astarithmetic "$bdir"/"$output_tidal_modulated" -h1 set-img img "$threshold_for_bright_pix" / set-norm_img \
            norm_img norm_img "$threshold_for_bright_pix" gt "$threshold_for_bright_pix" where --output="$bdir"/"$output_tidal_image_normalized"
            
            #I multiply the image of the stream with a larger central value by the constant surface brightness tidal tail
            output_tidal_image2=tidal_image2_"$ii".fits
            astarithmetic "$bdir"/"$stream_const" -h1 "$bdir"/"$output_tidal_image_normalized" -h1 x --output="$bdir"/"$output_tidal_image2"
            
            #boosting the flux to match the desired output
            flux_in_tidal_tail=$(aststatistics -h1 "$bdir"/"$output_tidal_image2" --sum)
            observed_mag=$(astarithmetic -q $flux_in_tidal_tail log10 -2.5 x $zeropoint +)
            boosting_factor=$(astarithmetic -q 10 $observed_mag $mag_stream - 0.4 x pow)
            output_tidal_image3=tidal_image3_"$ii".fits
            astarithmetic "$bdir"/"$output_tidal_image2" -h1 $boosting_factor x --output="$bdir"/"$output_tidal_image3"
            
            #Convolving by the PSF, both the galaxy and the tidal tail
    	    galaxy_convolved=galaxy_convolved_"$ii"_"$zz"_"$band".fits
            astconvolve "$bdir"/"$galaxy"              --kernel="$psf" -h1 --khdu 0 --domain=spatial --output="$bdir"/"$galaxy_convolved"
            stream_convolved=stream_convolved_"$ii"_"$zz"_"$band".fits
            astconvolve "$bdir"/"$output_tidal_image3" --kernel="$psf" -h1 --khdu 0 --domain=spatial --output="$bdir"/"$stream_convolved"
            
            #Converting to mag/arcsec^2
            galaxy_convolved_SB_units=galaxy_convolved_SB_units_"$ii"_"$zz"_"$band".fits
            astarithmetic "$bdir"/"$galaxy_convolved" $zeropoint counts-to-mag --output="$bdir"/"$galaxy_convolved_SB_units"
            stream_convolved_SB_units=stream_convolved_SB_units_"$ii"_"$zz"_"$band".fits
            astarithmetic "$bdir"/"$stream_convolved" $zeropoint counts-to-mag --output="$bdir"/"$stream_convolved_SB_units"
            
            #adding the cosmological dimming
            cosmological_dimming=$(astarithmetic -q 1 $zz + log10 7.5 x)
            #==applying it
            galaxy_convolved_SB_units_w_cosmo_dim=galaxy_convolved_SB_units_"$ii"_"$zz"_"$band"_with_cosmo_dim.fits
            astarithmetic "$bdir"/"$galaxy_convolved_SB_units" -h1 set-img img $cosmological_dimming + --output="$bdir"/"$galaxy_convolved_SB_units_w_cosmo_dim"
            stream_convolved_SB_units_w_cosmo_dim=stream_convolved_SB_units_"$ii"_"$zz"_"$band"_with_cosmo_dim.fits
            astarithmetic "$bdir"/"$stream_convolved_SB_units" -h1 set-img img $cosmological_dimming + --output="$bdir"/"$stream_convolved_SB_units_w_cosmo_dim"
            
            #Obtaining images to become labels because they display pixels 3sigma above the noise
            SB_limit=$(astarithmetic -q $sigma_over_noise log10 -2.5 x $bckg_mag +)
            stream_pix_above_surf_bright_limit=stream_pix_above_surf_bright_limit_"$ii"_"$zz"_"$band".fits
            astarithmetic "$bdir"/"$stream_convolved_SB_units_w_cosmo_dim" $SB_limit lt --output="$bdir"/"$stream_pix_above_surf_bright_limit"    
            
            #moving the images back to counts
            galaxy_convolved_w_cosmo_dim=galaxy_convolved_"$ii"_"$zz"_"$band"_with_cosmo_dim.fits
            astarithmetic "$bdir"/"$galaxy_convolved_SB_units_w_cosmo_dim" $zeropoint mag-to-counts --output="$bdir"/"$galaxy_convolved_w_cosmo_dim"
            stream_convolved_w_cosmo_dim=stream_convolved_"$ii"_"$zz"_"$band"_with_cosmo_dim.fits
            astarithmetic "$bdir"/"$stream_convolved_SB_units_w_cosmo_dim" $zeropoint mag-to-counts --output="$bdir"/"$stream_convolved_w_cosmo_dim"
        
            #I sum the final tidal tail to the galaxy's image
            galaxy_and_stream_convolved=galaxy_and_stream_convolved_"$ii"_"$zz"_"$band".fits
            astarithmetic "$bdir"/"$galaxy_convolved_w_cosmo_dim" -h1 "$bdir"/"$stream_convolved_w_cosmo_dim" -h1 2 sum --output="$bdir"/"$galaxy_and_stream_convolved"
            
            #Adding Poison noise -- requires bckg_mag
            #astarithmetic "$bdir"/galaxy_and_tidal_tail_convolved.fits $bckg_mag $zeropoint mag-to-counts mknoise-poisson --output="$bdir"/galaxy_and_tidal_tail_convolved_noised_"$ii".fits
            
            #Adding Gaussian noise according to the assumed background magnitude
            if [ $flag_gaussian_noise = "True" ]
            then
                galaxy_and_stream_convolved_with_noise=galaxy_and_stream_convolved_with_noise_"$ii"_"$zz"_"$band".fits
                bckg_in_counts=$(astarithmetic -q 10 $bckg_mag $zeropoint - -0.4 x pow)
                astarithmetic "$bdir"/"$galaxy_and_stream_convolved" "$bckg_in_counts" mknoise-sigma --output="$bdir"/"$galaxy_and_stream_convolved_with_noise"
            fi 
            
            #Creating a file with the main characteristics of each tidal stream
    	    width_feature=$(astarithmetic -q $end_angle $ini_angle -)
            #formatting the variables for the output catalog (LC_NUMERIC="en_US.UTF-8" for it to understand that . corresponds to the decimal point)
            zz_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $zz)
    	    mag_bulge_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $mag_bulge)
    	    re_bulge_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $re_bulge)
            re_bulge_pix_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $re_bulge_pix)
    	    ar_bulge_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $ar_bulge)
    	    pa_bulge_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $pa_bulge)
    	    mag_disk_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $mag_disk)
    	    re_disk_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $re_disk)
            re_disk_pix_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $re_disk_pix)
    	    ar_disk_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $ar_disk)
    	    pa_disk_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $pa_disk)
    	    mag_stream_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $mag_stream)
            rr_stream_pix_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $rr_stream_pix)
            width_stream_pix_cat=$(LC_NUMERIC="en_US.UTF-8" printf "%5.2f" $width_stream_pix) 
            echo "$ii $zz_cat $band $mag_bulge_cat $re_bulge_cat $re_bulge_pix_cat $ar_bulge_cat $pa_bulge_cat $mag_disk_cat $re_disk_cat $re_disk_pix_cat $ar_disk_cat $pa_disk_cat $mag_stream_cat $rr_stream_pix_cat $width_stream_pix_cat $azimuthal_width $ar_stream $pa_stream $ini_angle $end_angle $width_feature" >> "$cat_output" 
        
        done
        
    done

done
