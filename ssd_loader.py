import os
import numpy as np
from config import Settings
from PIL import Image
import pandas as pd
from rotate_ssd import rotate_ssd
import pickle


class Command:
    participant_id = 'P0'
    run_id = 'R0'
    timestamp = 0
    ACID = 'AC0000'
    command = 'HDG000'
    type = 'XXX'
    value = 0


def ssd_loader(settings):
    filelist = os.listdir(settings.ssd_folder)
    filelist = [file for file in filelist if '.png' in file]
    filelist.sort()
    print('Number of SSDs:', len(filelist))
    ssd_stack = []
    actions = pd.DataFrame(columns=['PARTICIPANT_ID', 'RUN_ID', 'TIME', 'ACID', 'TYPE', 'VALUE'])
    for i_file, filename in enumerate(filelist):
        if i_file % 100 == 0: print('SSD no.: {}/{}'.format(i_file, len(filelist)))
        command = Command
        """ TEXT PART """
        if 'SPD' in filename or 'HDG' in filename or 'DCT' in filename or 'TOC' in filename:
            command_split = filename.split('-')
            command.participant_id, \
            command.run_id, \
            command.timestamp, \
            command.ACID, \
            command.command = command_split[0:6]
            if len(command_split) > 6:
                print('Second command ignored:', filename)

        else:
            print('No SPD, HDG, DCT, or TOC command:', filename)
            continue

        command.timestamp = float(command.timestamp[1:])
        command.type = command.command[0:3]
        if command.type == 'HDG' or command.type == 'SPD':
            command.value = int(command.command[3:])
        else:
            command.value = 'N/A'
        actions.loc[i_file] = [command.participant_id, command.run_id, command.timestamp,
                               command.ACID, command.type, command.value]

        """ IMAGE PART """
        # TODO: Crop bottom half of image.
        ssd = Image.open(settings.ssd_folder + '/' + filename)
        if settings.convert_to_greyscale:
            ssd = ssd.convert("L")
        if settings.rotate_upwards:
            ssd = rotate_ssd(ssd, command)  # point speed vector up.
            if settings.crop_top:
                print('Only top half is taken into account')

        ssd = ssd.resize(settings.ssd_import_size, Image.NEAREST)
        filepath = 'data/all_ssd_edit/' + filename
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ssd.save(filepath)
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255
    # dimensions: (sample_num, x_size, y_size, number of color bands)
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], settings.ssd_import_size[0], settings.ssd_import_size[1], 3)

    actions.TIME.astype('float')
    actions.set_index('TIME', inplace=True)

    pickle.dump(actions, open(settings.data_folder + 'actions.p', "wb"))
    pickle.dump(ssd_stack, open(settings.data_folder + 'SSDs.p', "wb"))
    print('{} SDDs and actions saved to pickle.'.format(len(actions)))

    # returns (samples, dimension1, dimension2, 1) array with images.
    # and corresponding DF with actions
    return ssd_stack, actions


if __name__ == "__main__":
    settings = Settings
    ssd_loader(settings)
