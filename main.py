from config import settings
from serialize_data import serialize_data
from process_data import create_dataframes, analyse_commands, analyse_conflicts
from plot_data import plot_commands, plot_traffic
from ssd_loader import ssd_loader
from strategy_trainer import ssd_trainer
import pickle

def main():

    try:
        all_data = pickle.load(open(settings.data_folder + 'all_dataframes_3.p', "rb"))
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

        print('Start plotting')
        plot_commands()
        # plot_traffic()

        print('Start loading SSDs')
        all_dataframes = ssd_loader()

    print('Start training the neural network')

    settings.iteration_name = 'test1'
    ssd_trainer(all_data)

    settings.iteration_name = 'test2'
    ssd_trainer(all_data)


if __name__ == "__main__":

    main()
