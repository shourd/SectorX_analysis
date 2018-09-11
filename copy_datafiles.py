"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

import os
from pathlib import Path
from shutil import copyfile
# This script collects the data files from the available folders and puts it into a single folder

# Source and destination paths
source_path = '/Users/sjoerdvanrooijen/Dropbox/Thesis/SectorX/MSim/logdata/sjoerd_converging'
destination_path = '/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data'

# File extensions that need to be copied
file_extensions = [".xml", ".msg"]


def main():
    """ MAIN FUNCTION """
    print("Source path: \t\t..\\" + source_path.split("\\", 4)[-1])
    print("Destination path:\t..\\" + destination_path.split("\\", 4)[-1])
    print("File extentions:\t" + "".join(file_extensions).replace(".", " .").strip(), end="\n\n")
    input("Press enter to run...")
    print("")

    # Loop through source folder
    for subdir, dirs, files in os.walk(source_path):

        # For each file
        for file in files:
        
            # If the file has the wanted extension
            if os.path.splitext(file)[1] in file_extensions:
                copyDataFiles(subdir, file)


def copyDataFiles(subdir, file):
    ''' Copy data function'''
    # Create new file name with expertise level added in the name
    new_filename = os.path.basename(Path(subdir).parents[0]) + "_" + os.path.basename(file)

    # Copy file to new destination with new file name
    print("Copy " + file + " -> " + new_filename + "...", end="", flush=True)
    copyfile(os.path.join(subdir, file), os.path.join(destination_path, new_filename))
    print("Done!")


if __name__ == "__main__":
    ''' If the python script is run by itself'''
    main()
    print("")
