from config import settings
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import seaborn as sns
sns.set()

def make_radar_plot(df=None):
    # Set data
    if df is None:
        df = pd.read_csv('{}/test_scores.csv'.format(settings.output_dir))
        df.set_index('Unnamed: 0', inplace=True)
        df.index.name = 'participant'

    # number of variable
    categories = ['P{}'.format(participant_number) for participant_number in df.index]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.50, 0.75, 1], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=7)
    plt.ylim(0, 1)

    # Ind1
    values = df.iloc[:, 0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[0])
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values = df.iloc[:, 1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[1])
    ax.fill(angles, values, 'r', alpha=0.1)

    # Ind3
    values = df.iloc[:, 2].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[2])
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('MCC for testset using a general model')
    plt.savefig('{}/spinplot.png'.format(settings.output_dir), bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    make_radar_plot()
