from PIL import Image
import pickle
from config import Settings


def rotate_ssd(ssd_image, command):
    settings = Settings
    all_dataframes = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))

    df_traffic = all_dataframes['traffic']
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


    ssd_image = ssd_image.rotate(hdg, resample=Image.NEAREST, expand=0)

    return ssd_image
