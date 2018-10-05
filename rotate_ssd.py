from PIL import Image
import pickle
from config import Settings


def rotate_ssd(ssd_image, time, ACID):
    settings = Settings
    all_dataframes = pickle.load(open('../' + settings.data_folder + 'all_dataframes.p', "rb"))
    # TODO: remove sort and include in process file
    df_traffic = all_dataframes['traffic'].sort_index()
    df_commands = all_dataframes['commands'].sort_index()
    print(df_commands.columns)

    df_traffic_run = df_traffic.loc[('P1', 'R1')]

    traffic_timestamps = df_traffic_run[
        (df_traffic.participant_id == participant_id) & (df_traffic.run_id == run_id)].timestamp.unique()
    traffic_timestamps = [float(timestamp) for timestamp in traffic_timestamps]

    print('command time',time)
    df_traffic_line = df_traffic_run[
        (df_traffic_run.ACID == ACID)
        # (df_traffic_run.timestamp == time)
    ]
    print(df_traffic_line.to_string())
    # TODO: Make the traffic timestamp and command timestamp coincide.
    # command.timestamp_traffic = command.timestamp - 1  # command always taken at previous state
    # while command.timestamp_traffic not in traffic_timestamps:
    #     command.timestamp_traffic -= 1
    print('test')

    return ssd_image