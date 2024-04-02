import os
import shutil

paths_images_train = ["../segmentation_training/galaxies_train_VIS/", "../segmentation_training/galaxies_train_NISP_H/","../segmentation_training/galaxies_train_NISP_J/","../segmentation_training/galaxies_train_NISP_Y/"]
paths_masks_train = ["../segmentation_training/masks_train_VIS/", "../segmentation_training/masks_train_NISP_H/", "../segmentation_training/masks_train_NISP_J/", "../segmentation_training/masks_train_NISP_Y/"]

paths_images_test = ["../segmentation_training/galaxies_test_VIS/", "../segmentation_training/galaxies_test_NISP_H/","../segmentation_training/galaxies_test_NISP_J/","../segmentation_training/galaxies_test_NISP_Y/"]
paths_masks_test = ["../segmentation_training/masks_test_VIS/", "../segmentation_training/masks_test_NISP_H/", "../segmentation_training/masks_test_NISP_J/", "../segmentation_training/masks_test_NISP_Y/"]

for path_images_test in paths_images_test:
    if not os.path.exists(path_images_test):
        os.mkdir(path_images_test)

for path_masks_test in paths_masks_test:
    if not os.path.exists(path_masks_test):
        os.mkdir(path_masks_test)       

split_ratio_test = 0.2

def get_galaxy_number(galaxy_name:str):
    return os.path.basename(galaxy_name).split('_')[4]

def get_galaxy_magnitude(galaxy_name:str):
    return os.path.basename(galaxy_name).split('_')[5]

def move_files(source_dir, dest_dir, mask_source_dir, mask_dest_dir, percentage):
    files = os.listdir(source_dir)
    num_files_to_move = int(len(files) * percentage)
    files_to_move = files[:num_files_to_move]
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(source_path, dest_path)
        print(f"Moved fits file: {file_name}")
        # Move also corresponding mask
        numero_galaxia = get_galaxy_number(file_name)
        magnitud_galaxia = get_galaxy_magnitude(file_name)
        mask_file_name = "mask_"+str(numero_galaxia)+"_"+str(magnitud_galaxia)+".png"
        mask_source_path = os.path.join(mask_source_dir, mask_file_name)
        mask_dest_path = os.path.join(mask_dest_dir, mask_file_name)
        shutil.move(mask_source_path, mask_dest_path)
        print(f"Moved mask file: {mask_file_name}")


def split_datasets():
    # Move a fraction of the dataset to test
    for path_images_train, path_images_test, path_masks_train, path_masks_test in zip(paths_images_train, paths_images_test, paths_masks_train, paths_masks_test):
        move_files(path_images_train, path_images_test, path_masks_train, path_masks_test, split_ratio_test)



