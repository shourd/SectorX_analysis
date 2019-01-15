import os
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import settings


def plot_commands(all_dataframes):
    sns.set()  # reset all settings.
    sns.set('paper', 'whitegrid',
            rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
                'xtick.labelsize': 8, 'ytick.labelsize': 8},
            font='Times New Roman')

    os.makedirs(os.path.dirname(settings.data_folder + 'figures/extra'), exist_ok=True)
    df_commands = all_dataframes['commands'].reset_index()

    df_commands = df_commands[df_commands.type != 'N/A']
    df_commands = df_commands[df_commands.type != 'TOC']
    df_commands = df_commands[df_commands.ssd_id != 'N/A']

    # plot_traffic(all_dataframes)
    # return
    #
    # """ COUNT OF ALL COMMANDS """
    # fig, ax1 = plt.subplots(1, 1, figsize=settings.figsize_article)
    #
    # sns.countplot(data=df_commands, x='participant_id', hue='run_id', ax=ax1, palette='Blues')
    # ax1.legend_.set_title('Run')
    # ax1.set_xlabel('Particpant')
    # ax1.set_ylabel('Command count')
    # plt.savefig(settings.data_folder + 'figures/command_count_run.pdf', bbox_inches='tight')
    # # plt.savefig(settings.data_folder + 'figures/command_count_run.pgf', bbox_inches='tight')
    # plt.close()
    #
    # """ TYPE """
    # fig, ax2 = plt.subplots(1, 1, figsize=settings.figsize_article)
    # sns.countplot(data=df_commands[df_commands.type != 'N/A'], x='participant_id', hue='type', ax=ax2, palette='Blues')
    # ax2.legend_.set_title('Type')
    # ax2.set_xlabel('Participant')
    # ax2.set_ylabel('Command count')
    #
    # plt.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4, title='Type')
    #
    # plt.savefig(settings.data_folder + 'figures/command_count_type.pdf', bbox_inches='tight')
    # # plt.savefig(settings.data_folder + 'figures/command_count_type.pgf', bbox_inches='tight')
    # plt.close()
    #
    # """ Effect of the SSD """
    # fig, (ax1) = plt.subplots(1, 1, figsize=settings.figsize1)
    # sns.countplot(data=df_commands, x='type', hue='SSD', hue_order=['OFF', 'ON'], ax=ax1, palette='Blues')
    # # fig.suptitle('Command Count per Run')
    # ax1.set_xlabel('Command type')
    # plt.legend(['OFF', 'ON'], loc='lower right', bbox_to_anchor=(1, 1), ncol=2, title='SSD')
    # plt.savefig(settings.data_folder + 'figures/SSD_count.pdf', bbox_inches='tight')
    # plt.close()

    """ DIRECTION AND CONTROL PREFERENCE """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=settings.figsize3)
    data_direction = df_commands[(df_commands.direction != 'N/A')]
    data_direction = data_direction[(data_direction.direction != 'revert')]
    data_direction_spd = data_direction[data_direction.type == 'SPD']
    data_direction_hdg = data_direction[data_direction.type == 'HDG']
    data_geometry = df_commands[df_commands.preference != 'N/A']
    data_geometry = data_geometry[data_geometry.type.isin(['HDG', 'SPD'])]
    sns.countplot(data=data_direction_spd, x='participant_id', hue='direction',
                  hue_order=['decrease', 'increase'], ax=ax1)
    sns.countplot(data=data_direction_hdg, x='participant_id', hue='direction',
                  hue_order=['left', 'right'], ax=ax2)
    sns.countplot(data=data_geometry, x='participant_id', hue='preference', ax=ax3)
    fig.suptitle('Command preferences')
    ax1.set_title('Direction (Speed)')
    ax2.set_title('Direction (Heading)')
    ax3.set_title('Geometry')
    ax1.set_xlabel('Particpant ID')
    ax2.set_xlabel('Participant ID')
    plt.savefig(settings.data_folder + 'figures/command_preferences.pdf', bbox_inches='tight')
    plt.close()

    """ DIRECTION """
    data_direction = df_commands[(df_commands.direction != 'N/A')]
    data_direction_hdg = data_direction[data_direction.type == 'HDG']
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.countplot(data=data_direction_hdg, x='participant_id', hue='direction', hue_order=['left', 'right'], ax=ax,
                  palette='Blues')
    ax.legend_.set_title('Direction')
    ax.set_xlabel('Participant')
    ax.set_ylabel('HDG command count')
    ax.set_ylim([0, 100])
    plt.legend(['Left', 'Right'], loc='lower right', bbox_to_anchor=(1, 1), ncol=4, title='Direction')

    plt.savefig(settings.data_folder + 'figures/command_count_direction.pdf', bbox_inches='tight')
    # plt.savefig(settings.data_folder + 'figures/command_count_direction.pgf', bbox_inches='tight')
    plt.close()
    print('DONE')

    """ RELATIVE HEADING """
    hdg_commands = df_commands[df_commands.type == 'HDG']
    pd.options.mode.chained_assignment = None  # surprsses the copy warning from following statement:
    # hdg_commands.loc[:, 'hdg_rel'] = hdg_commands.hdg_rel.apply(lambda x: custom_round(x, base=20)).abs()
    hdg_commands.loc[:, 'hdg_rel'] = hdg_commands.hdg_rel.apply(lambda x: custom_round(x, base=20))
    pd.options.mode.chained_assignment = 'warn'

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize_article)
    sns.countplot(data=hdg_commands, x='hdg_rel', color=(146 / 255, 187 / 255, 211 / 255), ax=ax)
    ax.set_xlabel('Relative heading [deg]')
    ax.set_ylabel('HDG command count')
    ax.set_ylim([0, 250])
    plt.savefig(settings.data_folder + 'figures/command_count_value.pdf', bbox_inches='tight')
    # plt.savefig(settings.data_folder + 'figures/command_count_value.pgf', bbox_inches='tight')
    plt.close()
    print('DONE')

    """ COMMAND VALUES"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=settings.figsize2)
    spd_commands = df_commands[df_commands.type == 'SPD'].value
    spd_commands = pd.to_numeric(spd_commands)
    hdg_commands = df_commands[df_commands.type == 'HDG'].value
    hdg_commands = pd.to_numeric(hdg_commands)
    sns.distplot(spd_commands, bins=10, kde=False, ax=ax1)
    sns.distplot(hdg_commands, bins=36, kde=False, ax=ax2)
    ax1.set_title('SPD commands')
    ax1.set_xlabel('IAS [kts]')
    ax2.set_title('HDG commands')
    ax2.set_xlabel('HDG [deg]')
    fig.suptitle('Command Values')
    plt.savefig(settings.data_folder + 'figures/command_values.pdf', bbox_inches='tight')
    plt.close()

    # """ FACTOR PLOT COMMAND VALUES (ABSOLUTE HEADING) """
    # sns.catplot(x='value', col='participant_id', col_wrap=3, data=hdg_commands, kind='count', palette='muted')
    # plt.savefig(settings.data_folder + 'figures/extra/facet_rel_hdg.pdf', bbox_inches='tight')
    # plt.close()

    """ FACTOR PLOT COMMAND VALUES (RELATIVE HEADING) """
    hdg_commands = df_commands[df_commands.type == 'HDG']
    pd.options.mode.chained_assignment = None  # surprsses the copy warning from following statement:
    hdg_commands.loc[:, 'hdg_rel'] = hdg_commands.hdg_rel.apply(lambda x: custom_round(x, base=20)).abs()
    pd.options.mode.chained_assignment = 'warn'
    sns.catplot(x='hdg_rel', col='participant_id', col_wrap=3, data=hdg_commands, kind='count', palette='muted')
    plt.savefig(settings.data_folder + 'figures/extra/facet_rel_hdg_abs.pdf', bbox_inches='tight')
    plt.close()

    sns.set_style("darkgrid")
    df_commands.rename(columns={'participant_id': 'Participant'}, inplace=True)

    sns.set_style("darkgrid")
    ''' COMMAND TYPE TIMELINE '''
    df_commands_spd = df_commands[df_commands.type == 'SPD']
    g = sns.FacetGrid(df_commands_spd, col="Participant", col_wrap=3, palette='GnBu_d', aspect=1)
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    g.set(xlim=(0, 1200))
    g.axes[9].set_xlabel('Timestamp [s]')
    g.axes[10].set_xlabel('Timestamp [s]')
    g.axes[8].set_xlabel('Timestamp [s]')
    g.axes[0].set_ylabel('Run [-]')
    g.axes[3].set_ylabel('Run [-]')
    g.axes[6].set_ylabel('Run [-]')
    g.axes[9].set_ylabel('Run [-]')
    # plt.suptitle('Speed commands given')
    # g.fig.subplots_adjust(top=.9)
    plt.savefig(settings.data_folder + 'figures/extra/commands_over_time_spd.pdf', bbox_inches='tight')
    print('SPD timeline saved')

    df_commands_hdg = df_commands[df_commands.type == 'HDG']
    g = sns.FacetGrid(df_commands_hdg, col="Participant", col_wrap=3, palette='GnBu_d', aspect=1)
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    g.set(xlim=(0, 1200))
    g.axes[9].set_xlabel('Timestamp [s]')
    g.axes[10].set_xlabel('Timestamp [s]')
    g.axes[11].set_xlabel('Timestamp [s]')
    g.axes[0].set_ylabel('Run [-]')
    g.axes[3].set_ylabel('Run [-]')
    g.axes[6].set_ylabel('Run [-]')
    g.axes[9].set_ylabel('Run [-]')
    plt.savefig(settings.data_folder + 'figures/extra/commands_over_time_hdg.pdf', bbox_inches='tight')
    print('HDG timeline saved')

    df_commands_dct = df_commands[df_commands.type == 'DCT']
    g = sns.FacetGrid(df_commands_dct, col="Participant", col_wrap=3, palette='GnBu_d', aspect=1)
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    g.set(xlim=(0, 1200))
    g.axes[9].set_xlabel('Timestamp [s]')
    g.axes[10].set_xlabel('Timestamp [s]')
    g.axes[11].set_xlabel('Timestamp [s]')
    g.axes[0].set_ylabel('Run [-]')
    g.axes[3].set_ylabel('Run [-]')
    g.axes[6].set_ylabel('Run [-]')
    g.axes[9].set_ylabel('Run [-]')
    plt.savefig(settings.data_folder + 'figures/extra/commands_over_time_dct.pdf', bbox_inches='tight')
    print('DCT timeline saved')

    plt.close()


def custom_round(x, base=20):
    """ Rounds number to nearest base """
    return int(base * round(float(x) / base))


def plot_traffic(all_dataframes):
    # plot settings
    sns.set()
    sns.set_context("notebook")

    # """ HEATMAP """
    # df_traffic = all_dataframes['traffic'].reset_index()
    # ''' poging 1 '''
    # df_traffic = df_traffic.query("participant_id == 1")
    # df_traffic = df_traffic[(df_traffic.x_nm < 35)]
    # df_traffic = df_traffic[(df_traffic.x_nm > - 35)]
    # df_traffic = df_traffic[(df_traffic.y_nm < 25)]
    # df_traffic = df_traffic[(df_traffic.y_nm > - 35)]
    # f, ax = plt.subplots(figsize=(8, 8))
    # ax.set_aspect("equal")
    # sns.scatterplot(x=df_traffic.x_nm, y=df_traffic.y_nm, hue=df_traffic.run_id, ax=ax)
    # plt.savefig(settings.data_folder + 'figures/scatter.pdf', bbox_inches='tight')
    # plt.show()

    """ AIRCRAFT TYPE """

    main_flow_ACIDs = ['YG6251', 'XM1337', 'VS4694', 'UM6490', 'UG7514', 'EN5625', 'EF6739', 'AT5763', 'AH2854',
                       'AN1778']
    intruding_ACIDs = ['RA4743', 'SG3047', 'SM7071', 'PG4310', 'PA5424', 'RG3628', 'QM2514', 'OS2071', 'NA9895',
                       'OM3185']
    df_commands = all_dataframes['commands'].reset_index()
    num_commands_main = len(df_commands[df_commands.flow == 'main'])
    num_commands_intruding = len(df_commands[df_commands.flow == 'intruding'])
    print('Main:', num_commands_main)
    print('Intruding:', num_commands_intruding)

    df_commands.rename(columns={'participant_id': 'Participant'}, inplace=True)
    g = sns.catplot(hue='flow', hue_order=['main', 'intruding'], x='scenario',
                    col='Participant', col_wrap=3, data=df_commands, kind='count', height=3, aspect=1,
                    palette='Blues', legend=False)
    g.add_legend(title='Aircraft flow', labels=['Main', 'Intruding'])
    # g.set(xlim=(0, 1200))
    xlabel = 'Scenario'
    ylabel = 'Number of commands'
    g.axes[9].set_xlabel(xlabel)
    g.axes[10].set_xlabel(xlabel)
    g.axes[11].set_xlabel(xlabel)
    g.axes[0].set_ylabel(ylabel)
    g.axes[3].set_ylabel(ylabel)
    g.axes[6].set_ylabel(ylabel)
    g.axes[9].set_ylabel(ylabel)
    plt.savefig(settings.data_folder + 'figures/flow_preference.pdf', bbox_inches='tight')
    print('aircraft preference saved')
    plt.close()

    """ GEOMETRY """
    data_geometry = df_commands[df_commands.preference != 'N/A']
    data_geometry = data_geometry[data_geometry.type.isin(['HDG', 'SPD'])]
    g = sns.catplot(hue='preference', hue_order=['infront', 'behind'], x='scenario',
                    col='Participant', col_wrap=3, data=data_geometry, kind='count', height=3, aspect=1,
                    palette='Blues', legend_out=True, legend=False)
    # g._legend.set_title('Preference')
    # g._legend.set_label(['In front', 'Behind'])
    g.add_legend(title='Preference', labels=['In front', 'Behind'])
    xlabel = 'Scenario'
    ylabel = 'Number of commands'
    g.axes[9].set_xlabel(xlabel)
    g.axes[10].set_xlabel(xlabel)
    g.axes[11].set_xlabel(xlabel)
    g.axes[0].set_ylabel(ylabel)
    g.axes[3].set_ylabel(ylabel)
    g.axes[6].set_ylabel(ylabel)
    g.axes[9].set_ylabel(ylabel)
    plt.savefig(settings.data_folder + 'figures/geometry_preference.pdf', bbox_inches='tight')
    print('geometry preference saved')
    plt.close()


if __name__ == "__main__":
    # try:
    all_data = pickle.load(open(settings.data_folder + '181229_all_dataframes_crop_64.p', "rb"))
    #     print('Data loaded from Pickle')
    #     print('Start plotting')
    plot_commands(all_data)
    # except FileNotFoundError:
    #     print('Pickle all_dataframes_3.p not found. Please run process_data.py')

    # plot_traffic(all_data)
