""" CREATES RADARPLOT FROM DATAFRAME AND SAVES AS PDF/PGF """
import warnings

from config import settings
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import seaborn as sns

warnings.simplefilter(action='ignore', category=UserWarning)

def make_radar_plot(df=None, weights='overview', metric='MCC'):
    # plot settings
    sns.set()
    sns.set_context("notebook")   # smaller: paper
    sns.set('paper', 'darkgrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                  'xtick.labelsize': 8,
                                  'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    # sns.set_palette('Blues')

    # Set data
    if df is None:
        print('Provide DataFrame')
        return
    else:
        df.reset_index(inplace=True)

    participants = ['P{}'.format(participant_number) for participant_number in df.participant]
    N = len(participants)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(settings.figsize_article[0], 2.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], participants)

    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.50, 0.75, 1.00], ["0", "0.25", "0.50", "0.75", "1.00 {}".format(metric)], color="grey", size=7)
    plt.ylim(0, 1)
    # ax.xaxis.grid(True, color='grey', linestyle='-')

    if weights == 'overview':
        labels = ['General model', 'Individual model']
    else:
        labels = ['Type', 'Direction', 'Value']

    for i_column in range(len(labels)):
        values = df.iloc[:, i_column+1].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=labels[i_column])
        ax.fill(angles, values, 'b', alpha=0.1)

    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 1.2), ncol=2)
    if settings.save_as_pgf:
        plt.savefig('{}/test_scores/spinplot_{}_{}.pgf'.format(settings.output_dir, weights, metric), bbox_inches='tight')
    plt.savefig('{}/test_scores/spinplot_{}_{}.pdf'.format(settings.output_dir, weights, metric), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    make_radar_plot()
