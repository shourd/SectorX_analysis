import pickle
import time

import pandas as pd
import matplotlib

from config import settings
from plot_data import plot_commands
from plot_results import plot_results
from process_data import create_dataframes, analyse_commands
from serialize_data import serialize_data
from ssd_loader import ssd_loader
from strategy_trainer import ssd_trainer

# matplotlib.use('agg')  # fixes a multi-thread issue.

def main():

    try:
        all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
        print('Data loaded from pickle')
    except FileNotFoundError:
        print('Start serializing data')
        participant_list = serialize_data()

        print('Start processing data')
        all_dataframes = create_dataframes(participant_list)

        print('Start processing commands')
        all_dataframes = analyse_commands(all_dataframes)

        # print('Start processing conflicts')
        # all_dataframes = analyse_conflicts(participant_list)

        print('Start loading SSDs')
        all_data = ssd_loader(all_dataframes)

        print('Start plotting')
        plot_commands(all_data)
        # plot_traffic()

    print('Start training the neural network')

    # measure training time
    start_time = time.time()
    metrics_all_df = pd.DataFrame()

    for target_type in settings.target_types:
        settings.target_type = target_type

        for participant in settings.participants:
            settings.current_participant = participant
            participant_ids = [participant]

            for ssd_condition in settings.ssd_conditions:
                settings.ssd = ssd_condition
                settings.iteration_name = '{}_{}_{}_{}'.format(target_type, participant,
                                                                     settings.experiment_name,
                                                                  ssd_condition)
                print('------------------------------------------------')
                print('-- Start training:', settings.iteration_name)
                print('------------------------------------------------')

                """ TRAIN MODEL """
                metrics_iteration_df = ssd_trainer(all_data, participant_ids)

                """ SAVE TO DISK """
                if not metrics_iteration_df.empty:
                    metrics_all_df = metrics_iteration_df if metrics_all_df.empty else metrics_all_df.append(metrics_iteration_df)
                    metrics_all_df.to_csv(settings.output_dir + '/metrics_{}.csv'.format(settings.experiment_name))

    print('Train time: {} min'.format(round(int(time.time() - start_time) / 60), 1))
    plot_results(settings.experiment_name)


if __name__ == "__main__":
    main()
