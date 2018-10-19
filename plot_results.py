from config import settings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
sns.set()

def plot_results():

    experiment_name = 'geometry_normal'

    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()
    # results = pd.read_csv(settings.output_dir + '/metrics_onlydirection.csv').reset_index()

    print('Max MCC: ',round(results.MCC.max(),2))
    print('Max val_acc: ',round(results.val_acc.max(),2))
    print('Max F1 score: ',round(results.val_F1_score.max(),2))
    print('Max informedness: ',round(results.val_informedness.max(),2))

    results.rename(columns={'iteration_name': 'Augmentation'}, inplace=True)

    # palette = itertools.cycle(sns.color_palette())

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=settings.figsize4)

    sns.lineplot(x='epoch', y='MCC', data=results, hue='Augmentation', ax=ax1)
    ax1.set_ylim([-0.25, 1])
    ax1.set_ylabel('MCC')

    sns.lineplot(x='epoch', y='val_acc', data=results, hue='Augmentation', ax=ax2)
    ax2.set_ylim([0.4, 1])
    ax2.set_ylabel('Accuracy')

    sns.lineplot(x='epoch', y='val_informedness', hue='Augmentation', data=results, ax=ax3)
    ax3.set_ylim([0.4, 1])
    ax3.set_ylabel('Informedness')

    sns.lineplot(x='epoch', y='val_F1_score', hue='Augmentation', data=results, ax=ax4)
    ax4.set_ylim([0.4, 1])
    ax4.set_ylabel('F1 score')

    plt.savefig(settings.output_dir + '/results_not_converted.png', bbox_inches='tight')


if __name__ == '__main__':
    plot_results()
