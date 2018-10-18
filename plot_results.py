from config import settings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
sns.set()

def plot_results():

    experiment_name = 'not_converted'
    results = pd.DataFrame.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()

    print('Max MCC: ',results.MCC.max())
    print('Max val_acc: ',round(results.val_acc.max(),2))
    print('Max F1 score: ',results.val_F1_score.max())
    print('Max informedness: ',results.val_informedness.max())


    palette = itertools.cycle(sns.color_palette())

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=settings.figsize4)

    sns.lineplot(x='epoch', y='MCC', data=results, ax=ax1)
    ax1.set_ylim([-1, 1])

    sns.lineplot(x='epoch', y='val_acc', data=results, ax=ax2)
    ax2.set_ylim([0, 1])

    sns.lineplot(x='epoch', y='val_informedness', data=results, ax=ax3)
    ax3.set_ylim([0, 1])

    sns.lineplot(x='epoch', y='val_F1_score', data=results, ax=ax4)
    ax4.set_ylim([0, 1])

    plt.savefig('results_not_converted.png', bbox_inches='tight')


