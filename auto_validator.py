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
    settings.ssd_conditions = ['BOTH']
    target_types = ['type', 'direction', 'value']

    """ START EVALUATION"""
    validation_scores_df = pd.DataFrame()
    commands, ssd_stack = load_test_data()

    for validation_participant_id in participants:
        streep()
        print('Validation participant data:', validation_participant_id)
        streep()

        weights = validation_participant_id
        participant_scores_dict = {'type': 0, 'direction': 0, 'value': 0}

        for target_type in target_types:
            participant_scores_dict[target_type] = [evaluate_target_type(weights, validation_participant_id,
                                                                         target_type, ssd_stack, commands)[metric]]

        validation_scores_df = validation_scores_df.append(pd.DataFrame.from_dict(participant_scores_dict))

    # Finalize dataframe
    validation_scores_df.index = participants
    validation_scores_df.index.name = 'participant'
    validation_scores_df['average'] = validation_scores_df[target_types].mean(axis=1)
    validation_scores_df.to_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, 'auto_val'))

    print(validation_scores_df)

    return validation_scores_df


def plot_scores():
    df = pd.read_csv('{}/test_scores/test_scores_{}.csv'.format(settings.output_dir, 'auto_val'))

    mean_test_score = df.average.mean().round(3)
    print('Mean test score:', mean_test_score)

    sns.set('paper', 'darkgrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    sns.set_palette('Blues')
    df_melt = df.melt(id_vars='participant')
    fig, ax = plt.subplots(figsize=settings.figsize_article)
    plt.axhline(y=0.75, linewidth=1, color='k', linestyle='--')
    sns.barplot(data=df_melt, x='participant', y='value', hue='variable', ax=ax)

    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    ax.set_xlabel('Participant')
    plt.savefig('{}/test_scores/test_perf_participant.pdf'.format(settings.output_dir), bbox_inches='tight')
    plt.savefig('{}/test_scores/test_perf_participant.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # scores_df = main()
    plot_scores()
    # make_radar_plot(scores_df, weights='auto_val')