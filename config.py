# config.py

class Settings:
    # DATA LOCATIONS
    data_folder = 'data/all/'
    ssd_folder = 'data/all_ssd'
    output_dir = 'output'
    serialized_data_filename = "serialized_data.p"
    input_file = 'all_dataframes_3_half.p'

    # SERIALIZE AND PROCESS SETTINGS
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']

    # PLOT SETTINGS
    show_plots = False  # show plots when running scripts
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)
    figsize4 = (20, 5)

    # SSD IMPORT SETTINGS
    convert_to_greyscale = False
    convert_black_to_white = True
    rotate_upwards = True  # rotates the speedvector towards the north
    crop_top = True  # only possible when rotated upwards
    save_png_files = True
    ssd_import_size = (128, 128)
    ignore_PRV = True


    # CNN TRAIN SETTINGS
    experiment_name = 'repetition'
    # participants = ['all']
    # participants = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'all']
    participants = ['P1', 'P1', 'P1', 'P1', 'P1', 'P1', 'P1', 'P1', 'P1', 'P1']
    ssd = 'all' #'ON'  # 'OFF' , 'all'
    target_types = ['relative_heading']
    load_weights = False  #'relative_heading_all_weight_init'
    train_val_ratio = 0.75
    epochs = 20
    batch_size = 32  # 128
    steps_per_epoch = 5  #888 / 128
    rotation_range = 0  # the maximum degree of random rotation for data augmentation
    # num_classes = 2  # amount of resolution classes (2, 4, 6, or 12)
    save_model = True  # save model weights to disk
    freeze_layers = False
    reload_data = False
    csv_logger = False
    dropout_rate = 0.2
    class_names = []


    def __init__(self):
        width = self.ssd_import_size[0]
        if self.crop_top:
            height = int(self.ssd_import_size[1] / 2)
        else:
            height = int(self.ssd_import_size[1])

        if self.convert_to_greyscale:
            channels = 1
        else:
            channels = 3

        self.ssd_shape = (height, width, channels)

global settings
settings = Settings()

