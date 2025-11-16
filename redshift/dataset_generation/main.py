import subprocess
from nan_to_black import nan_to_black
from mask_creation import create_masks
from inserting_tidal_tails import insert_tidal_tails
from datasets_splitting import split_datasets
from nan_deleting import delete_nans

def run_gnu_astro_script():
    command = "bash create_mock_tidal_tails_multi_z_v2.sh"
    process = subprocess.Popen(command, shell=True)
    process.wait()

if __name__ == "__main__":

    #First, run GNUAstro bash script
    #run_gnu_astro_script()

    #Second, run nan to black python script
    #nan_to_black()

    #Third, generate labels with python
    #create_masks()

    #Fourth, run inserting tidal tails scripts
    #insert_tidal_tails()

    #Fifth, divide datasets in train and test
    split_datasets()

    delete_nans()
