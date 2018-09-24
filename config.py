# config.py


class Settings:
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    show_plots = False          # show plots when running scripts
    data_folder = 'data/P3/'
    serialized_data_filename = "serialized_data.p"
    processed_data_filename = 'processed_data.p'

