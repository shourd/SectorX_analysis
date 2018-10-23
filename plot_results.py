from config import settings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
sns.set()

def plot_results(experiment_name):

    results = pd.read_csv(settings.output_dir + '/metrics_{}.csv'.format(experiment_name)).reset_index()
    # results = pd.read_csv(settings.output_dir + '/metrics_onlydirection.csv').reset_index()

    p_list = []
    for iteration in list(results.iteration_name):
        _, _, p_no, _ = iteration.split('_')
        p_list.append(p_no)

    results.iteration_name = p_list


    print('Max MCC: ',round(results.MCC.max(),2))
    print('Max val_acc: ',round(results.val_acc.max(),2))
    print('Max F1 score: ',round(results.val_F1_score.max(),2))
    print('Max informedness: ',round(results.val_informedness.max(),2))

    results.rename(columns={'iteration_name': 'Iteration name'}, inplace=True)

    # palette = itertools.cycle(sns.color_palette())

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=settings.figsize4)

    sns.lineplot(x='epoch', y='MCC', data=results, hue='Iteration name', ax=ax1, legend='brief')
    # ax1.set_ylim([-0.25, 1])
    ax1.set_ylabel('MCC')

    # box = ax1.get_position()  # get position of figure
    # ax1.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    ax1.legend(loc='upper center', bbox_to_anchor=(2.3, 1.15), ncol=12)

    sns.lineplot(x='epoch', y='val_acc', data=results, hue='Iteration name', ax=ax2, legend=False)
    # ax2.set_ylim([0.4, 1])
    ax2.set_ylabel('Accuracy')

    sns.lineplot(x='epoch', y='val_informedness', hue='Iteration name', data=results, ax=ax3, legend=False)
    # ax3.set_ylim([0.4, 1])
    ax3.set_ylabel('Informedness')

    sns.lineplot(x='epoch', y='val_F1_score', hue='Iteration name', data=results, ax=ax4, legend=False)
    # ax4.set_ylim([0.4, 1])
    ax4.set_ylabel('F1 score')
    plt.savefig('{}/{}.png'.format(settings.output_dir, settings.experiment_name), bbox_inches='tight')


if __name__ == '__main__':
    # plot_results('dropout0')
    plot_results(settings.experiment_name)
