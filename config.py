# config.py
import numpy as np

class Settings:
    # DATA LOCATIONS
    data_folder = 'data/all/'
    ssd_folder = 'data/all_ssd'
    output_dir = 'output'
    serialized_data_filename = "serialized_data.p"
    input_file = '181101_all_dataframes_3.p'

    # SSD IMPORT SETTINGS
    convert_background = False
    remove_grey_noise = True
    rotate_upwards = True  # rotates the speed vector towards the north
    crop_top = True  # only possible when rotated upwards
    convert_to_greyscale = False
    ssd_import_size = (128, 128)
    save_png_files = True
    ignore_PRV = True
    export_file = '181101_all_dataframes_3.p'

    # CNN TRAIN SETTINGS
    experiment_name = 'DCTtest'
    repetitions = 5
    participants = np.arange(1,13,1)  # [1 .. 12]
    participants = np.arange(1, 7, 1)  # [1 .. 12]
    ssd_conditions = ['ON', 'OFF']
    target_types = ['geometry', 'type', 'direction', 'value']
    target_types = ['direction']
    load_weights = False # 'direction_all_full_experiment_pooling_rep15'
    train_val_ratio = 0.75
    epochs = 30
    batch_size = 32
    rotation_range = 0  # the maximum degree of random rotation for data augmentation
    save_model_structure = True  # save png of model structure to disk
    freeze_layers = False
    reload_data = False
    dropout_rate = 0.2

    #Callbacks
    callback_save_model = False  # save model weights to disk
    callback_tensorboard = True #output log data to tensorboard
    matthews_correlation_callback = True
    show_model_summary = False

    # SERIALIZE AND PROCESS SETTINGS
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    # columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']

    # PLOT SETTINGS
    show_plots = False  # show plots when running scripts
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)
    figsize4 = (20, 5)

    # inits
    current_participant = 'P0'
    current_repetition = 0
    class_names = []
    num_classes = 2
    skill_level = 'N/A'
    ssd = 'BOTH'
    # steps_per_epoch = 0

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

    def determine_skill_level(self):
        participant = self.current_participant
        if participant in np.arange(1,7,1):
            self.skill_level = 'novice'
        elif participant in np.arange(7,13,1):
            self.skill_level = 'intermediate'
        else:
            self.skill_level = 'all'

        return self.skill_level


global settings
settings = Settings()

