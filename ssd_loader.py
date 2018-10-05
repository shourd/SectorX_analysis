import os
import numpy as np
from config import Settings
from PIL import Image
import pandas as pd
from cnn.rotate_ssd import rotate_ssd


def ssd_loader(settings):
    filelist = os.listdir(settings.ssd_folder)
    filelist = [file for file in filelist if '.png' in file]
    print('Number of SSDs:', len(filelist))
    ssd_stack = []
    actions = pd.DataFrame(columns=['TIME', 'ACID', 'TYPE', 'VALUE'])
    for i_file, filename in enumerate(filelist):

        """ TEXT PART """
        if 'SPD' in filename or 'HDG' in filename or 'DCT' in filename:
            try:
                timestamp, ACID, command, _ = filename.split('-')
            except ValueError:
                print('error! Filename:', filename)
        else:
            continue

        timestamp = float(timestamp[1:])
        command_type = command[0:3]
        if command_type == 'HDG' or command_type == 'SPD':
            command_value = command[3:]
        else:
            command_value = 'N/A'
        actions.loc[i_file] = [timestamp, ACID, command_type, command_value]

        """ IMAGE PART """
        # TODO: Crop bottom half of image.
        # TODO: Rotate with speed vector pointing up.
        ssd = Image.open(settings.ssd_folder + '/' + filename)
        # ssd = ssd.convert("L")  # convert to greyscale
        ssd = ssd.resize(settings.ssd_import_size, Image.BILINEAR)
        ssd = rotate_ssd(ssd, timestamp, ACID)  # point speedvector up.
        # ssd.show()
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255
    # dimensions: (sample_num, x_size, y_size, number of color bands)
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], settings.ssd_import_size[0], settings.ssd_import_size[1], 3)

    actions.set_index('TIME', inplace=True)
    # actions.sort_index(inplace=True)

    # returns (samples, dimension1, dimension2, 1) array with images.
    return ssd_stack, actions


if __name__ == "__main__":
    settings = Settings
    ssd_loader(settings)
