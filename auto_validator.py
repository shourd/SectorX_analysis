""" TESTS ALL MODELS AGAINST ALL TEST DATA AND PLOTS 1 RADAR PLOT PER PARTICIPANT """

import numpy as np
import pandas as pd
import warnings

from config import settings
from cross_validator import evaluate_target_type, load_test_data, streep
from radar_plot import make_radar_plot

import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore', category=UserWarning)


def main():
    """ SETTINGS """
    participants = np.arange(1, 13, 1)  # 'all' or Participant ID (integer)
    # OUTPUT
    metric = 2  # 1 = accuracy, 2 = MCC
    # CONDITIONS

    target_types = ['type', 'direction', 'value']

    """ START EVALUATION"""
    validation_scores_df = pd.DataFrame()
    commands, ssd_stack = load_test_data()

    ssd_conditions = ['ON', 'OFF']
    # ssd_conditions = ['BOTH']

    for ssd_condition in ssd_conditions:
        settings.ssd_condition = ssd_condition

        for validation_participant_id in participants:
            streep()
            print('Validation participant data:', validation_participant_id)
            streep()

            weights = validation_participant_id

            participant_scores_dict = {'participant': validation_participant_id,
                                       'SSD': ssd_condition,
                                       'skill_level': 'novice' if validation_participant_id < 7 else 'intermediate'}

            for target_type in target_types:
                scores = evaluate_target_type(weights, validation_participant_id,
                                                                             target_type, ssd_stack,
                                                                             commands, ssd_condition)
                participant_scores_dict[target_type + '_acc'] = [scores[1]]
                participant_scores_dict[target_type + '_mcc'] = [scores[2]]
                # participant_scores_dict[target_type] = [evaluate_target_type(weights, validation_participant_id,
                #                                                              target_type, ssd_stack,
                #                                                              commands, ssd_condition)[metric]]

            validation_scores_df = validation_scores_df.append(pd.DataFrame.from_dict(participant_scores_dict))

    # Finalize dataframe
    # validation_scores_df.index = participants
    # validation_scores_df.index.name = 'participant'
    validation_scores_df['acc_mean'] = validation_scores_df[['type_acc', 'direction_acc', 'value_acc']].mean(axis=1)
    validation_scores_df['mcc_mean'] = validation_scores_df[['type_mcc', 'direction_mcc', 'value_mcc']].mean(axis=1)
    validation_scores_df.to_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, 'auto_ssd'))

    print(validation_scores_df)

    return validation_scores_df


def plot_scores():
    df = pd.read_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, 'auto'))

    print('Mean test score (MCC):', df.mcc_mean.mean().round(3))
    print('Mean test score (ACC):', df.acc_mean.mean().round(3))

    sns.set('paper', 'ticks', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8},
            font='Times New Roman')

    sns.set_palette('Blues')

    # """ PER PARTICIPANT """
    df = df[['participant', 'type_mcc', 'direction_mcc', 'value_mcc']]
    df.rename(columns={'type_mcc': 'Type', 'direction_mcc': 'Direction', 'value_mcc': 'Value'}, inplace=True)
    df_melt = df.melt(id_vars='participant')
    #
    # fig, ax = plt.subplots(figsize=settings.figsize_article)
    # # plt.axhline(y=0.75, linewidth=1, color='k', linestyle='--')
    # sns.boxplot(data=df_melt, x='participant', y='value', color=(146 / 255, 187 / 255, 211 / 255), saturation=1,
    #             linewidth=1, fliersize=2, ax=ax)
    # # plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    # ax.set_xlabel('Participant')
    # ax.set_ylabel('MCC')
    # plt.savefig('{}/test_scores/test_perf_participant.pdf'.format(settings.output_dir), bbox_inches='tight')
    # if settings.save_as_pgf:
    #     plt.savefig('{}/test_scores/test_perf_participant.pgf'.format(settings.output_dir), bbox_inches='tight')
    # plt.close()

    """ BARPLOT """
    fig, ax = plt.subplots(figsize=settings.figsize_article)
    sns.barplot(data=df_melt, x='participant', y='value', hue='variable', saturation=1, ax=ax, palette='Blues')
    sns.despine()
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, title='Abstraction Level')
    ax.set_xlabel('Participant')
    ax.set_ylabel('MCC')
    ax.set_ylim([0, 1])
    plt.savefig('{}/test_scores/test_perf_participant_bar.pdf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()

    # """ PER TYPE """
    # # df_melt = df.melt(id_vars='')
    # fig, ax = plt.subplots(figsize=settings.figsize_article)
    # # plt.axhline(y=0.75, linewidth=1, color='k', linestyle='--')
    # sns.boxplot(data=df_melt, x='variable', y='value', linewidth=0.5, fliersize=2, ax=ax)
    # sns.despine()
    # # plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    # ax.set_xlabel('Abstraction level')
    # ax.set_ylabel('MCC')
    # plt.savefig('{}/test_scores/test_perf_target_type.pdf'.format(settings.output_dir), bbox_inches='tight')
    # plt.close()


if __name__ == '__main__':

    # scores_df = main()
    plot_scores()
    # make_radar_plot(scores_df, weights='auto_val')
