import os
import pickle

import numpy as np
from PIL import Image

from config import settings


class Command:
    participant_id = 0
    run_id = 'R0'
    timestamp = 0
    ACID = 'AC0000'
    command = 'HDG000'
    type = 'XXX'
    value = 0


def ssd_loader(dataframes=None):
    if dataframes is None:
        dataframes = pickle.load(open(settings.data_folder + 'all_dataframes_2.p', "rb"))

    df_traffic = dataframes['traffic']
    df_commands = dataframes['commands']
    df_commands_edit = df_commands.reset_index()
    df_commands_edit['ssd_id'] = 'N/A'

    filelist = os.listdir(settings.ssd_folder)
    filelist = [file for file in filelist if '.png' in file]
    filelist.sort()
    print('Number of SSDs:', len(filelist))

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
            print('ERROR:', filename)


        """ IMAGE PART """
        ssd = Image.open(settings.ssd_folder + '/' + filename)

        ssd = edit_ssd(ssd, command, df_traffic)

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
    ssd_stack = ssd_stack.reshape(ssd_stack.shape[0], settings.ssd_shape[0], settings.ssd_shape[1], settings.ssd_shape[2])

    """ SAVING AND RETURNING """
    df_commands = df_commands_edit.set_index(['participant_id', 'run_id', 'i_command'])
    dataframes['commands'] = df_commands
    dataframes['ssd_images'] = ssd_stack

    pickle.dump(dataframes, open(settings.data_folder + settings.export_file, "wb"))
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
            command.participant_id = int(command.participant_id[1:])  # convert PX to X

    else:
        print('No SPD, HDG, DCT, or TOC command:', filename)
        command = None

    return command


def edit_ssd(ssd, command, df_traffic):

    if settings.rotate_upwards:
        ssd = rotate_ssd(ssd, command, df_traffic)  # point speed vector up.
        if settings.crop_top:
            crop_fraction = 0.5
            ssd = ssd.crop(box=(0, 0, ssd.width, ssd.height*crop_fraction))

    if settings.convert_background or settings.remove_grey_noise:
        ssd = transform_colors(ssd)

    if settings.convert_to_greyscale:
        ssd = ssd.convert("L")

    # resize and crop
    if settings.crop_top:
        ssd = ssd.resize((settings.ssd_import_size[0], int(settings.ssd_import_size[1]*crop_fraction)), Image.NEAREST)
    if not settings.crop_top:
        ssd = ssd.resize(settings.ssd_import_size, Image.NEAREST)

    return ssd

def transform_colors(ssd):
    """
    :param ssd PIL Image
    :return: ssd PIL Image with transformed colors
    """

    from_color = (255, 255, 255)
    if settings.convert_background:
        new_background_color = (0, 0, 0)
    else:
        new_background_color = (255, 255, 255)

    ssd_array = np.array(ssd)  # "data" is a height x width x R x G x B array
    red, green, blue = ssd_array.T # new format: RGB x HW

    if settings.convert_background:
        white_areas = (red == from_color[0]) & (blue == from_color[1]) & (green == from_color[2])
        ssd_array[white_areas.T] = new_background_color

    if settings.remove_grey_noise:
        grey_areas = (red == blue) & (red == green) & (red < 255) & (red > 0)
        orange_areas = (red > green) & (green > blue)
        red_areas = (red > 210) & (green == blue) & (red != green)
        blue_areas = (blue > red) & (blue > green) & (blue != 255)  # do not remove the blue spee

        ssd_array[grey_areas.T] = new_background_color
        ssd_array[orange_areas.T] = (255, 212, 160)
        ssd_array[red_areas.T] = (255, 0, 0)
        ssd_array[blue_areas.T] = new_background_color

    ssd = Image.fromarray(ssd_array)

    return ssd


def rotate_ssd(ssd_image, command, df_traffic):

    df_traffic_run = df_traffic.loc[(command.participant_id, command.run_id)]
    traffic_timestamps = df_traffic_run.timestamp.unique()

    command.timestamp_traffic = command.timestamp - 1  # command always taken at previous state
    while command.timestamp_traffic not in traffic_timestamps:
        command.timestamp_traffic -= 1

    try:
        hdg = df_traffic_run[
            (df_traffic_run.ACID == command.ACID) &
            (df_traffic_run.timestamp == command.timestamp_traffic)
        ].hdg_deg.iloc[0]
    except IndexError:
        print('heading lookup error')
        print(command.participant_id, command.run_id)
        print('Timestamp command', command.timestamp_traffic)
        print('command ACID', command.ACID)
        print('stop')
        print(df_traffic_run.to_string())
        print(df_traffic_run.ACID.unique())

    ssd_image = ssd_image.rotate(hdg, resample=Image.NEAREST, expand=False)

    return ssd_image


if __name__ == "__main__":
    ssd_loader()
