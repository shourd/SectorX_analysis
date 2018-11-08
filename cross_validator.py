import keras
import numpy as np
import pandas as pd
import pickle
import glob

from command_predictor import load_model
from config import settings
from strategy_trainer import prepare_training_set
from radar_plot import make_radar_plot


def main(model_weights='all'):
    """ SETTINGS """
    # VALIDATION DATA
    participant_ids = np.arange(1, 13, 1)
    # OUTPUT
    metric = 2  # 1 = accuracy, 2 = MCC
    # CONDITIONS
    settings.ssd_conditions = ['BOTH']
    target_types = ['type', 'direction', 'value']

    """ START EVALUATION"""
    validation_scores_df = pd.DataFrame()
    commands, ssd_stack = load_test_data()

    streep()
    print('MODEL TO BE EVALUATED:', model_weights)

    for validation_participant_id in participant_ids:
        streep()
        print('Validation participant data:', validation_participant_id)
        streep()

        participant_scores_dict = {'type': 0, 'direction': 0, 'value': 0}

        for target_type in target_types:
            participant_scores_dict[target_type] = [evaluate_target_type(model_weights, validation_participant_id,
                                                                         target_type, ssd_stack, commands)[metric]]

        validation_scores_df = validation_scores_df.append(pd.DataFrame.from_dict(participant_scores_dict))

    # Finalize dataframe
    validation_scores_df.index = participant_ids
    validation_scores_df.index.name = 'participant'
    validation_scores_df['average'] = validation_scores_df[target_types].mean(axis=1)
    validation_scores_df.to_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, model_weights))

    return validation_scores_df


def evaluate_target_type(weights, validation_participant_id, target_type, x_data, y_data):
    run_ids = ['R4']
    x_data, y_data = prepare_training_set(x_data, y_data,
                                          participant_ids=[validation_participant_id],
                                          target_type=target_type,
                                          run_ids=run_ids)

    model = load_model('model_architecture_3class')
    filepath_weights = glob.glob('{}/weights/{}_{}*.hdf5'.format(settings.output_dir, target_type, weights))
    if len(filepath_weights) > 1: print('MULTIPLE WEIGHTS FOR SAME MODEL DETECTED')
    model.load_weights(filepath_weights[0])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', 'matthews_correlation'])

    test_score = model.test_on_batch(x_data, y_data, sample_weight=None)

    return test_score  # Val MCC over entire set.


def load_test_data():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()
    commands_df = commands_df[commands_df.ssd_id != 'N/A']
    ssd_stack = all_data['ssd_images']

    return commands_df, ssd_stack


def combine_score_dfs(weights_list):
    df_baseline = pd.read_csv('{}/test_scores/test_scores_all.csv'.format(settings.output_dir))
    mcc_general_model = df_baseline.average

    mcc_personal_model_array = np.empty(len(weights_list)-1)
    for weights in weights_list:

        if weights is 'all':
            continue

        df_temp = pd.read_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, weights))
        mcc_personal_model_array[weights-1] = df_temp[df_temp.participant == weights].average.iloc[0]
    mcc_personal_model = pd.DataFrame(mcc_personal_model_array, columns=['mcc_personal_model'])

    combined_df = pd.concat([mcc_general_model, mcc_personal_model], axis=1)
    combined_df.index = np.arange(1, 13, 1)
    combined_df.index.name = 'participant'

    return combined_df


def streep():
    print('--------------------------------------------------')


if __name__ == '__main__':
    # MODEL WEIGHTS:
    weights_list = ['all', 1, 2]  # 'all' or Participant ID (integer)
    # settings.run_ids = ['R4']

    # START
    for weights in weights_list:
        scores_df = main(weights)
        make_radar_plot(scores_df, weights)

    df = combine_score_dfs(weights_list)
    make_radar_plot(df)
