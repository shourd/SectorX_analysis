# facet_radar_plot.py
import warnings
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings

sns.set()
sns.set_context("paper")

warnings.simplefilter(action='ignore', category=UserWarning)


def main():
    plot_type = 'normal'  # or 'average'
    df_combined = pd.read_csv('{}/test_scores/test_scores_combined.csv'.format(settings.output_dir))

    # plot settings
    sns.set()
    sns.set_context("notebook")  # smaller: paper
    sns.set('paper', 'darkgrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8, "pgf.rcfonts": False},
            font='Times New Roman')

    df_combined.type.loc[df_combined.type < 0] = 0  # convert negative numbers to zero for better looking plots.

    """ SINGLE PARTICIPANT PLOT """
    participant_id = 1
    participant_df = df_combined.loc[df_combined.model_participant == participant_id]
    make_spider(model_participant_id=participant_id, df=participant_df, radar_type=plot_type, single_plot=True)
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 1.2), ncol=3)
    plt.tight_layout()
    plt.savefig('{}/test_scores/radar_plot_P{}.pdf'.format(settings.output_dir, participant_id), bbox_inches='tight')
    plt.close()
    print('Single plot DONE')

    """ FACET PLOT """
    plt.figure(figsize=(8, 10))

    for participant_id in range(1,13):
        participant_df = df_combined.loc[df_combined.model_participant == participant_id]
        make_spider(model_participant_id=participant_id, df=participant_df, radar_type=plot_type)
        if participant_id == 3:
            plt.legend(loc='lower left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('{}/test_scores/facet_radar_plot_{}.pdf'.format(settings.output_dir, plot_type), bbox_inches='tight')
    print('Facet Plot DONE')


def make_spider(model_participant_id, df, radar_type, single_plot=False):

    validation_participants = ['P{}'.format(participant_number) for participant_number in df.participant]
    N = len(validation_participants)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    if single_plot:
        fig, ax = plt.subplots(figsize=(settings.figsize_article[0], 3))
        ax = plt.subplot(111, polar=True)
    else:
        ax = plt.subplot(4, 3, model_participant_id, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], validation_participants)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1], ["0.25", "0.50", "0.75", "1.00 MCC"], color="grey", size=7)
    plt.ylim(0, 1)

    if radar_type == 'average':
        labels = ['average']
    else:
        labels = ['type', 'direction', 'value']


    for i_column in range(len(labels)):
        values = df.loc[:, [labels[i_column]]].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=labels[i_column])
        ax.fill(angles, values, 'b', alpha=0.1)

    if not single_plot:
        plt.title('P{} model'.format(model_participant_id), y=1.2, fontweight='bold')


if __name__ == '__main__':
    main()
