""" TESTS ALL MODELS AGAINST ALL TEST DATA AND PLOTS 1 RADAR PLOT PER PARTICIPANT """

import glob
import pickle

import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from command_predictor import load_model
from config import settings
from radar_plot import make_radar_plot
from strategy_trainer import prepare_training_set
from confusion_matrix_script import get_confusion_metrics


def main(model_weights='all', weights_folder=''):
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
            # participant_scores_dict[target_type] = [evaluate_target_type(model_weights, validation_participant_id,
            #                                                              target_type, ssd_stack, commands)[metric]]

            scores = evaluate_target_type(model_weights, validation_participant_id,
                                          target_type, ssd_stack,
                                          commands, weights_folder=weights_folder)

            participant_scores_dict[target_type + '_acc'] = [scores[1]]  # acc
            participant_scores_dict[target_type] = [scores[2]]  # mcc

        validation_scores_df = validation_scores_df.append(pd.DataFrame.from_dict(participant_scores_dict))

    # Finalize dataframe
    validation_scores_df.index = participant_ids
    validation_scores_df.index.name = 'participant'
    validation_scores_df['average'] = validation_scores_df[target_types].mean(axis=1)
    validation_scores_df.to_csv('{}/test_scores/test_scores_{}_{}.csv'.format(settings.output_dir, weights_folder, model_weights))

    return validation_scores_df


def evaluate_target_type(weights, validation_participant_id, target_type, x_data, y_data, ssd='BOTH', weights_folder=''):
    run_ids = ['R4']  # Important: only used Run 4 as test data. All models have been trained on Runs 1-3
    x_data, y_data = prepare_training_set(x_data, y_data,
                                          participant_ids=[validation_participant_id],
                                          target_type=target_type,
                                          run_ids=run_ids)

    model = load_model('paper_seed2')
    if 'general' in weights_folder:
        weights_folder = 'weights_{}'.format(weights_folder)
    else:
        weights_folder = 'weights'
    filepath_weights = glob.glob('{}/{}/{}_{}_*_{}_*.hdf5'
                                     .format(settings.output_dir, weights_folder, target_type, weights, ssd))
    if len(filepath_weights) > 1: print('MULTIPLE WEIGHTS FOR SAME MODEL DETECTED')
    if len(filepath_weights) == 0: print('NO WEIGHTS DETECTED: {}/weights/{}_{}_*_{}_*.hdf5'.format(settings.output_dir, target_type, weights, ssd))
    model.load_weights(filepath_weights[0])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', 'matthews_correlation'])

    y_pred = model.predict(x_data)
    y_pred_rounded = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_data, axis=1)

    get_confusion_metrics(y_test, y_pred_rounded, save_figure=True, participant=validation_participant_id, target_type=target_type)

    test_score = model.test_on_batch(x_data, y_data, sample_weight=None)

    return test_score  # Val MCC over entire set.


def load_test_data():
    all_data = pickle.load(open(settings.data_folder + settings.input_file, "rb"))
    commands_df = all_data['commands'].reset_index()
    commands_df = commands_df[commands_df.ssd_id != 'N/A']
    ssd_stack = all_data['ssd_images']

    return commands_df, ssd_stack


def combine_score_dfs(metric):
    df_baseline = pd.read_csv('{}/test_scores/test_scores_general_model.csv'.format(settings.output_dir))
    mcc_general_model = df_baseline.mean_mcc
    acc_general_model = df_baseline.mean_acc

    # mcc_personal_model_array = np.empty(len(model_weights_list))
    # combined_df = pd.DataFrame()
    # for model_weights in model_weights_list:
    #
    #     df_temp = pd.read_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, model_weights))
    #     df_temp['model_participant'] = model_weights
    #     combined_df = combined_df.append(df_temp)
    #
    #     if model_weights != 'all':
    #         mcc_personal_model_array[model_weights - 1] = df_temp[df_temp.participant == model_weights].average.iloc[0]
    #
    # combined_df.to_csv('{}/test_scores/test_scores_combined.csv'.format(settings.output_dir))
    # print('Combined dataframe saved')

    test_scores_auto_df = pd.read_csv('{}/test_scores/test_scores_auto.csv'.format(settings.output_dir))
    mcc_personal_model = test_scores_auto_df.mcc_mean
    acc_personal_model = test_scores_auto_df.acc_mean

    summary_df = pd.concat([mcc_general_model, mcc_personal_model, acc_general_model, acc_personal_model], axis=1)
    summary_df.index = np.arange(1, 13, 1)
    summary_df.index.name = 'participant'
    summary_df.columns = ['general_mcc', 'individual_mcc', 'general_acc', 'individual_acc']
    summary_df['delta_mcc'] = summary_df.individual_mcc - summary_df.general_mcc
    summary_df['delta_acc'] = summary_df.individual_acc - summary_df.general_acc

    summary_df.to_csv('{}/test_scores/test_scores_summary.csv'.format(settings.output_dir))
    print('Summary dataframe saved')

    plot_delta_values(summary_df)

    if metric == 'mcc':
        summary_df = summary_df.loc[:, ['general_mcc', 'individual_mcc']]
    elif metric == 'acc':
        summary_df = summary_df.loc[:, ['general_acc', 'individual_acc']]

    return summary_df


def streep():
    print('--------------------------------------------------')


def plot_delta_values(df):

    print('General MCC:', list(df.general_mcc.round(3)))
    print('Individual MCC:', list(df.individual_mcc.round(3)))
    streep()
    print('General ACC:', list(df.general_acc.round(3)))
    print('Individual ACC:', list(df.individual_acc.round(3)))

    df_delta = df.loc[:,['delta_mcc', 'delta_acc']]
    df_melt = df_delta.reset_index().melt(id_vars='participant')

    sns.set('paper', 'whitegrid',
            rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8},
            font='Times New Roman')

    fig, ax = plt.subplots(figsize=settings.figsize_article)
    sns.barplot(data=df_melt, x='participant', y='value', hue='variable', ax=ax, palette='Blues')
    plt.ylabel('Performance difference')
    plt.xlabel('Participant')
    plt.ylim([-0.2, 0.2])
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, title='')
    plt.savefig('{}/test_scores/delta_general_individual.pdf'.format(settings.output_dir), bbox_inches='tight')

    df_mcc = df.loc[:, ['general_mcc', 'individual_mcc']]
    df_mcc.rename(columns={'general_mcc': 'General', 'individual_mcc': 'Individual'}, inplace=True)
    df_mcc = df_mcc.reset_index().melt(id_vars='participant')

    df_acc = df.loc[:, ['general_acc', 'individual_acc']]
    df_acc.rename(columns={'general_acc': 'General', 'individual_acc': 'Individual'}, inplace=True)
    df_acc = df_acc.reset_index().melt(id_vars='participant')

    """ FINAL COMPARISON GENERAL INDIVIDUAL """
    sns.set('paper', 'whitegrid',
            rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8},
            font='Times New Roman')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=settings.figsize_article)
    g = sns.boxplot(data=df_mcc, x='variable', y='value', palette='Blues',
                    linewidth=0.5, fliersize=2, ax=ax1)
    sns.despine()
    ax1.set_ylabel('MCC')
    ax1.set_xlabel('Model')
    ax1.set_ylim([0.3, 1])

    g = sns.boxplot(data=df_acc, x='variable', y='value', palette='Blues',
                    linewidth=0.5, fliersize=2, ax=ax2)
    sns.despine()
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Model')
    plt.ylim([0.3, 1])

    plt.tight_layout()
    plt.savefig('{}/test_scores/general_individual_comp.pdf'.format(settings.output_dir), bbox_inches='tight')
    print('general_individual_comp.pdf Saved')


if __name__ == '__main__':

    # MODEL WEIGHTS:
    weights_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 'all' or Participant ID (integer)

    # for weights in weights_list:
    #     scores_df = main(weights)
    #     make_radar_plot(scores_df, weights)

    metric = 'mcc'
    df = combine_score_dfs(metric=metric)
    make_radar_plot(df, metric=metric)
    print('Overview radar plot saved')
    print('Results plotted')
