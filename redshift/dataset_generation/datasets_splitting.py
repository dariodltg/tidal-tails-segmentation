import os
import shutil

paths_images_train = ["segmentation_training/galaxies_train_VIS/", "segmentation_training/galaxies_train_NISP_H/","segmentation_training/galaxies_train_NISP_J/","segmentation_training/galaxies_train_NISP_Y/"]
paths_masks_train = ["segmentation_training/masks_train_VIS/", "segmentation_training/masks_train_NISP_H/", "segmentation_training/masks_train_NISP_J/", "segmentation_training/masks_train_NISP_Y/"]

paths_images_test = ["segmentation_training/galaxies_test_VIS/", "segmentation_training/galaxies_test_NISP_H/","segmentation_training/galaxies_test_NISP_J/","segmentation_training/galaxies_test_NISP_Y/"]
paths_masks_test = ["segmentation_training/masks_test_VIS/", "segmentation_training/masks_test_NISP_H/", "segmentation_training/masks_test_NISP_J/", "segmentation_training/masks_test_NISP_Y/"]


split_ratio_test = 0.2

def move_files(source_dir, dest_dir, percentage):
    files = os.listdir(source_dir)
    num_files_to_move = int(len(files) * percentage)
    files_to_move = files[:num_files_to_move]
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.move(source_path, dest_path)
        print(f"Moved file: {file_name}")

def split_datasets():
    # Move a fraction of the dataset images to test
    for path_images_train, path_images_test in zip(paths_images_train, paths_images_test):
        move_files(path_images_train, path_images_test, split_ratio_test)
    
    # Move a fraction of the dataset masks to test
    for path_masks_train, path_masks_test in zip(paths_masks_train, paths_masks_test):
        move_files(path_masks_train, path_masks_test, split_ratio_test)



