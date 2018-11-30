
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings
from performance_consistency_comparison import label_point
import numpy as np


def main():
    num_samples_df = pd.read_csv(settings.output_dir + '/{}/{}_num_samples.csv'.format(settings.experiment_name, settings.experiment_name))
    num_samples_df = num_samples_df.loc[:, ['type', 'direction', 'value']]
    performance_df = pd.read_csv(settings.output_dir + '/test_scores/test_scores_auto.csv')
    performance_df.drop('Unnamed: 0', axis=1, inplace=True)

    combined_df = num_samples_df.merge(performance_df, on='participant')
    print(combined_df.to_string())

    # Comparison plot
    sns.set('paper', 'darkgrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                     'xtick.labelsize': 8,
                                     'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})

    participants = pd.Series(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12'])

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(settings.figsize_article[0], 6))

    sns.regplot(x='type', y='type_mcc', robust=True, ci=None, data=combined_df, ax=ax)
    # label_point(combined_df.type, combined_df.type_mcc, participants, plt.gca())
    ax.set_xlabel('Number of training samples: Type')
    ax.set_ylabel('Type MCC')
    ax.set_ylim([0, 1])

    sns.regplot(x='direction', y='direction_mcc', robust=True, ci=None, data=combined_df, ax=ax2)
    # label_point(combined_df.direction, combined_df.direction_mcc, combined_df.participant, plt.gca())
    ax2.set_xlabel('Number of training samples: Direction')
    ax2.set_ylabel('Direction MCC')
    ax2.set_ylim([0, 1])

    sns.regplot(x='value', y='value_mcc', robust=True, ci=None, data=combined_df, ax=ax3)
    # label_point(combined_df.value, combined_df.value_mcc, participants, plt.gca())
    ax3.set_xlabel('Number of training samples: Value')
    ax3.set_ylabel('Value MCC')
    ax3.set_ylim([0, 1])

    plt.tight_layout()
    #

    plt.savefig('{}/{}/{}.pdf'.format(settings.output_dir, settings.experiment_name, 'samples_performance'),
                bbox_inches='tight')
    if settings.save_as_pgf:
        plt.savefig('{}/{}/{}.pgf'.format(settings.output_dir, settings.experiment_name, 'samples_performance'), bbox_inches='tight')

    print('Regplots plotted and saved.')

if __name__ == '__main__':
    main()