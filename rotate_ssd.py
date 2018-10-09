from PIL import Image
import pickle
from config import Settings


def rotate_ssd(ssd_image, command):
    settings = Settings
    all_dataframes = pickle.load(open(settings.data_folder + 'all_dataframes.p', "rb"))
    # TODO: remove sort and include in process file and convert timestamp column to floats
    df_traffic = all_dataframes['traffic'].sort_index()
    df_traffic.timestamp = df_traffic.timestamp.astype('float')
    df_commands = all_dataframes['commands'].sort_index()

    df_traffic_run = df_traffic.loc[(command.participant_id, command.run_id)]

    traffic_timestamps = df_traffic_run.timestamp.unique()
    # traffic_timestamps = [float(timestamp) for timestamp in traffic_timestamps]

    command.timestamp_traffic = command.timestamp - 1  # command always taken at previous state
    while command.timestamp_traffic not in traffic_timestamps:
        command.timestamp_traffic -= 1

    hdg = df_traffic_run[
        (df_traffic_run.ACID == command.ACID) &
        (df_traffic_run.timestamp == command.timestamp_traffic)
    ].hdg_deg.iloc[0]

    ssd_image = ssd_image.rotate(hdg, resample=Image.NEAREST, expand=0)

    return ssd_image
