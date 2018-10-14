# config.py


class Settings:
    # DATA LOCATIONS
    data_folder = 'data/all/'
    ssd_folder = 'data/all_ssd'
    output_dir = 'output'
    serialized_data_filename = "serialized_data.p"
    processed_data_filename = 'all_dataframes_3.p'

    # serialize and process settings
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    show_plots = False          # show plots when running scripts
    columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)

    # SSD IMPORT SETTINGS
    convert_to_greyscale = False
    rotate_upwards = True  # rotates the speedvector towards the north
    crop_top = True  # only possible when rotated upwards
    save_png_files = False
    ssd_import_size = (128, 128)
    ignore_PRV = True

    # def ssd_shape(self):
    #     width = self.ssd_import_size[0]
    #     if self.crop_top:
    #         height = self.ssd_import_size[1]/2
    #     else:
    #         height = self.ssd_import_size
    #
    #     if self.convert_to_greyscale:
    #         channels = 1
    #     else:
    #         channels = 3
    #
    #     ssd_shape = (width, height, channels)
    #
    #     return ssd_shape

    def __init__(self):
        width = self.ssd_import_size[0]
        if self.crop_top:
            height = int(self.ssd_import_size[1] / 2)
        else:
            height = self.ssd_import_size

        if self.convert_to_greyscale:
            channels = 1
        else:
            channels = 3

        self.ssd_shape = (width, height, channels)

    # CNN TRAIN SETTINGS
    # learning_rate = 0.01
    train_val_ratio = 0.8
    # model_name = 'model'
    epochs = 1
    batch_size = 128  # 128
    steps_per_epoch = 10  #888 / 128
    rotation_range = 0  # the maximum degree of random rotation for data augmentation
    num_classes = 3  # amount of resolution classes (2, 4, 6, or 12)
    randomize_fraction = 0  # Randomize percentage of samples to simulate human randomness
    save_model = False  # save model weights to disk
    reload_data = False
    use_dropout = False



