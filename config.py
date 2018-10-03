# config.py


class Settings:
    caution_time = 120          # orange conflict
    warning_time = 60           # red conflict
    distance_to_sector = 50     # distance at which to include aircraft in relevant aircraft list
    show_plots = False          # show plots when running scripts
    data_folder = 'data'
    serialized_data_filename = "serialized_data.p"
    processed_data_filename = 'processed_data.p'
    columns = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12']
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)

