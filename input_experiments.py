import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings


def main():
    warnings.simplefilter(action='ignore', category=UserWarning)
    sns.set()
    sns.set_context("notebook")  # smaller: paper
    sns.set('paper', 'whitegrid', rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.titlesize': 10,
                                      'xtick.labelsize': 8,
                                      'ytick.labelsize': 8, "pgf.rcfonts": False})
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    sns.set_palette('Blues')


    results = pd.read_csv(settings.output_dir + '/input_experiments.csv')
    print(results)

    results_melt = results.melt(id_vars='Input type')

    results_melt = results_melt.rename(columns={'variable': 'Level', 'value': 'MCC'})

    fig, ax = plt.subplots(figsize=settings.figsize_article)
    sns.barplot(x='Input type', y='MCC', hue='Level', data=results_melt, ax=ax)
    ax.set_ylim([0,1])
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4)
    plt.savefig('{}/input_experiments.pdf'.format(settings.output_dir), bbox_inches='tight')
    plt.savefig('{}/input_experiments.pgf'.format(settings.output_dir), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
