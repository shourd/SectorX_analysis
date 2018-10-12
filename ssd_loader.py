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


def ssd_loader(dataframes, settings):
    df_traffic = dataframes['traffic']
    df_commands = dataframes['commands']
    df_commands_edit = df_commands.reset_index()
    df_commands_edit['ssd_id'] = 'N/A'

    filelist = os.listdir(settings.ssd_folder)
    filelist = [file for file in filelist if '.png' in file]
    filelist.sort()
    print('Number of SSDs:', len(filelist))

    # actions = pd.DataFrame(columns=['PARTICIPANT_ID', 'RUN_ID', 'TIME', 'ACID', 'TYPE', 'VALUE'])

    ssd_stack = []
    ssd_id = 0
    for i_file, filename in enumerate(filelist):
        if i_file % 100 == 0: print('SSD no.: {}/{}'.format(i_file, len(filelist)))

        command = generate_command(filename)
        if command is None:
            continue

        df_commands_edit.loc[
            (df_commands_edit.participant_id == command.participant_id) &
            (df_commands_edit.run_id == command.run_id) &
            (df_commands_edit.timestamp == command.timestamp), 'ssd_id'] = ssd_id

        if ssd_id not in list(df_commands_edit.ssd_id):
            print('nee', filename)


        """ IMAGE PART """
        ssd = Image.open(settings.ssd_folder + '/' + filename)

        ssd = edit_ssd(ssd, command, df_traffic, settings)

        if settings.save_png_files:
            filepath = 'data/all_ssd_edit/' + filename
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            ssd.save(filepath)

        ssd = np.array(ssd)
        ssd_stack.append(ssd)
        ssd_id += 1 # only increase id when image is added.


    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255

    # dimensions: (sample_num, height, width px, number of color bands /channels)
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], settings.ssd_shape[1], settings.ssd_shape[0], settings.ssd_shape[2])

    """ SAVING AND RETURNING """
    df_commands = df_commands_edit.set_index(['participant_id', 'run_id', 'i_command'])
    dataframes['commands'] = df_commands
    dataframes['ssd_images'] = ssd_stack

    pickle.dump(dataframes, open(settings.data_folder + 'all_dataframes_3.p', "wb"))
    print('-----------------------------------------------------------------')
    print('{} SDDs saved to pickle.'.format(len(ssd_stack)))
    print('{} SSDs filtered out'.format(i_file - ssd_id))

    return dataframes


def generate_command(filename):
    command = Command

    if 'SPD' in filename or 'HDG' in filename or 'DCT' in filename or 'TOC' in filename:
        if 'PRV' in filename and settings.ignore_PRV:
            print('PRV command ignored', filename)
            command = None
        elif 'TOC' in filename:
            # print('TOC command ignored', filename)
            command = None
        else:
            command_split = filename.split('-')
            command.participant_id, \
            command.run_id, \
            command.timestamp, \
            command.ACID, \
            command.command = command_split[0:5]
            if len(command_split) > 6:
                print('Second command ignored:', filename)

            command.timestamp = float(command.timestamp[1:])
            command.type = command.command[0:3]
            if command.type == 'HDG' or command.type == 'SPD':
                command.value = int(command.command[3:])
            else:
                command.value = 'N/A'

    else:
        print('No SPD, HDG, DCT, or TOC command:', filename)
        command = None

    return command


def edit_ssd(ssd, command, df_traffic, settings):
    if settings.convert_to_greyscale:
        ssd = ssd.convert("L")
    if settings.rotate_upwards:
        ssd = rotate_ssd(ssd, command, df_traffic)  # point speed vector up.
        if settings.crop_top:
            crop_fraction = 0.5
            ssd = ssd.crop(box=(0, 0, settings.ssd_import_size[0], settings.ssd_import_size[0]*crop_fraction))

    if settings.crop_top:
        ssd = ssd.resize((settings.ssd_import_size[0], settings.ssd_import_size[1]*crop_fraction), Image.NEAREST)
    if not settings.crop_top:
        ssd = ssd.resize(settings.ssd_import_size, Image.NEAREST)

    return ssd


if __name__ == "__main__":
    settings = Settings()
    # all_dataframes = pickle.load(open(settings.data_folder + 'all_dataframes.p', "rb"))
    # print(all_dataframes['commands'].to_string())

    all_dataframes = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))
    # print(all_dataframes['commands'].to_string())
    ssd_loader(all_dataframes, settings)
