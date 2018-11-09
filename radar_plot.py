from config import settings
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import seaborn as sns
sns.set()

def make_radar_plot(df=None, weights='overview'):
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

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], participants)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=7)
    plt.ylim(0, 0.75)

    if weights == 'overview':
        labels = ['General model', 'Personal model']
    else:
        labels = ['Type', 'Direction', 'Value']

    for i_column in range(len(labels)):
        values = df.iloc[:, i_column+1].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=labels[i_column])
        ax.fill(angles, values, 'b', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    if weights == 'overview':
        plt.title('MCC of personal models compared to general model', y=1.1)
    else:
        plt.title('P{} model tested on all individual test sets'.format(weights), y=1.1)
    plt.savefig('{}/test_scores/spinplot_{}.png'.format(settings.output_dir, weights), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    make_radar_plot()
