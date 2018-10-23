from config import settings
from serialize_data import serialize_data
from process_data import create_dataframes, analyse_commands, analyse_conflicts
from plot_data import plot_commands, plot_traffic
from ssd_loader import ssd_loader
from strategy_trainer import ssd_trainer
import pickle
import pandas as pd
from plot_results import plot_results

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

    metrics_all = pd.DataFrame()
    for target_type in settings.target_types:
        settings.target_type = target_type
        settings.load_weights = False

        for participant in settings.participants:
            print('------------------------------------------------')
            print('-- Start training:', participant)
            print('------------------------------------------------')
            settings.iteration_name = '{}_{}_{}'.format(target_type, participant, settings.experiment_name)
            participant_ids = [participant]
            metrics_run = ssd_trainer(all_data, participant_ids)
            metrics_run.index.name = 'epoch'
            if metrics_all.empty:
                metrics_all = metrics_run
            else:
                metrics_all = metrics_all.append(metrics_run)

            metrics_all.to_csv(settings.output_dir + '/metrics_{}.csv'.format(settings.experiment_name))

    plot_results(settings.experiment_name)

if __name__ == "__main__":

    main()
