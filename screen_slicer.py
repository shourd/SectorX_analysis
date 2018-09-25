""" Imports screenshots from MSim and extracts the SSD image """

from PIL import Image
from config import Settings
import numpy as np
import os
import pickle


def screen_slicer(size, input_folder, output_folder):
    files = os.listdir(settings.data_folder + input_folder)
    png_files = [file for file in files if '.png' in file]
    print('Number of files:', len(png_files))
    # xml_files = sorted(xml_files, key=lambda x: int(x[-1]))
    # png_files = [file+'.xml' for file in png_files]
    ssd_stack = []
    for fname in png_files:
        # fname = glob('{}*'.format(fname))[0]  # take only first entry
        screenshot = Image.open(settings.data_folder + input_folder + '/' + fname)
        # ssd = ssd.convert("L")  # convert to greyscale
        screenshot_width, screenshot_height = screenshot.size
        left = 0
        top = screenshot_height - 128
        right = 128
        bottom = screenshot_height
        ssd = screenshot.crop((left, top, right, bottom))
        # ssd.show()
        ssd = ssd.resize(size, Image.BILINEAR)
        filepath = settings.data_folder + input_folder + '/' + output_folder + '/' + fname[6:]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ssd.save(filepath)
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack) # dimensions: (sample_num, x_size, y_size, number of color bands)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255

    return ssd_stack


if __name__ == "__main__":
    settings = Settings
    size = (128, 128)
    print('Start loading SSDs...')
    output_folder = 'crop'
    participants = 3
    runs = 4
    for p in range(participants):
        print('Participant: ', p + 1)
        for r in range(runs):
            settings.data_folder = 'data/P{}/'.format(p + 1)
            input_folder = 'P{}_run{}'.format(p + 1, r + 1)
            ssd_stack = screen_slicer(size, input_folder, output_folder)

            # save data
            pickle_name = 'SSDs_{}.pickle'.format(input_folder)
            pickle.dump(ssd_stack, open(settings.data_folder + pickle_name, "wb"))
