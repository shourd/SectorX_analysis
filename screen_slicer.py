""" Imports screenshots from MSim and extracts the SSD image """
from glob import glob
from PIL import Image
from config import Settings
import numpy as np
from os import listdir
import pickle


def screen_slicer(size, folder):
    # filelist = sorted(filelist, key=os.path.getmtime) # to sort on time modified
    # filelist = glob('{}screendumps/*.png'.format(folder))
    files = listdir(settings.data_folder+folder)
    png_files = [file for file in files if '.png' in file]
    print('Number of files:', len(png_files))
    # xml_files = sorted(xml_files, key=lambda x: int(x[-1]))
    # png_files = [file+'.xml' for file in png_files]
    ssd_stack = []
    for fname in png_files:
        # fname = glob('{}*'.format(fname))[0]  # take only first entry
        screenshot = Image.open(settings.data_folder+folder+'/'+fname)
        # ssd = ssd.convert("L")  # convert to greyscale
        screenshot_width, screenshot_height = screenshot.size
        left = 0
        top = screenshot_height - 128
        right = 128
        bottom = screenshot_height
        ssd = screenshot.crop((left, top, right, bottom))
        # ssd.show()
        ssd = ssd.resize(size, Image.BILINEAR)
        ssd.save(settings.data_folder+folder+'/crop/'+fname[6:])
        ssd = np.array(ssd)
        ssd_stack.append(ssd)

    ssd_stack = np.array(ssd_stack)
    ssd_stack = ssd_stack.astype('float32')
    ssd_stack /= 255
    # dimensions: (sample_num, x_size, y_size, number of color bands)
    return ssd_stack


if __name__ == "__main__":
    settings = Settings
    size = (128, 128)
    ssd_stack = screen_slicer(size, folder='screendumps')

    # save data
    pickle.dump(ssd_stack, open(settings.data_folder+"ssd_all.pickle", "wb"))
    print('Data saved to disk')
