import subprocess
from redshift.dataset_generation.nan_to_black import nan_to_black
from redshift.dataset_generation.mask_creation import create_masks
from redshift.dataset_generation.inserting_tidal_tails import insert_tidal_tails
from redshift.dataset_generation.datasets_splitting import split_datasets
if __name__ == "__main__":
    #First, run GNUAstro bash script

    command = "bash create_mock_tidal_tails_multi_z_v2.sh"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    #Second, run nan to black python script
    nan_to_black()

    #Third, generate labels with python
    create_masks()

    #Fourth, run inserting tidal tails scripts
    insert_tidal_tails()

    #Fifth, divide datasets in train and test
    split_datasets()
