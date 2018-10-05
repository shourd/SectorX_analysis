# config.py


class Settings:
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    show_plots = False          # show plots when running scripts
    data_folder = 'data/all/'
    serialized_data_filename = "serialized_data.p"
    processed_data_filename = 'processed_data.p'
    columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)
    # SSD IMPORT SETTINGS
    ssd_folder = 'crop_all'
    ssd_import_size = (64, 64)
    # CNN TRAIN SETTINGS
    train_val_ratio = 0.8
    model_name = 'model'
    epochs = 10
    batch_size = 128
    steps_per_epoch = 12 #888 / 128
    rotation_range = 0  # the maxium degree of random rotation for data augmentation
    num_classes = 3  # amount of resolution classes (2, 4, 6, or 12)
    randomize_fraction = 0  # Randomize percentage of samples to simulate human randomness
    save_model = False  # save trained model to disk
    reload_data = False
    output_dir = 'output'
