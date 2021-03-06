# config.py
import numpy as np


class Settings:
    # DATA LOCATIONS
    data_folder = 'data/all/'
    ssd_folder = 'data/all_ssd'
    output_dir = 'output'
    serialized_data_filename = "serialized_data.p"
    input_file = '181121_all_dataframes_crop_64.p'

    # SSD IMPORT SETTINGS
    convert_background = False  # 'white' / 'black' / False
    remove_grey_noise = False
    rotate_upwards = True  # rotates the speed vector towards the north
    crop_top = True  # only possible when rotated upwards
    convert_to_greyscale = False
    ssd_import_size = (64, 64)
    save_png_files = True
    ignore_PRV = True
    export_file = '181229_all_dataframes_crop_64.p'

    # CNN TRAIN SETTINGS
    seed = 1
    experiment_name = 'paper_seed2'
    repetitions = 5  # number of folds (for cross-validation)
    participants = np.arange(1, 13, 1)  # [1 .. 12]
    participants = [6]
    run_ids = ['R1', 'R2', 'R3']
    ssd_conditions = ['ON', 'OFF', 'BOTH']
    ssd_conditions = ['BOTH']
    target_types = ['type', 'direction', 'value']
    limit_data = True  # only applicable when participant = 'all'!
    load_weights = False  # 'direction_all_full_experiment_pooling_rep15'
    epochs = 30
    batch_size = 32
    rotation_range = 0  # the maximum degree of random rotation for data augmentation
    freeze_layers = False
    reload_data = False
    dropout_rate = 0.2

    #Callbacks
    callback_save_model = False  # save model weights to disk
    callback_tensorboard = False #output log data to tensorboard
    matthews_correlation_callback = True
    show_model_summary = False
    save_model_structure = True  # save png of model structure to disk

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
    figsize_article = (5, 2)
    figsize_article_high = (5, 3)


    # inits
    current_participant = 'P0'
    current_repetition = 0
    class_names = []
    num_classes = 2
    skill_level = 'N/A'
    ssd = 'BOTH'
    target_type = 'None'
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

    #


global settings
settings = Settings()
