import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import config
import pickle


def plot_commands(all_dataframes):
    df_commands = all_dataframes['commands'].reset_index()
    # df_traffic = all_dataframes['traffic'].reset_index()

    df_commands = df_commands[df_commands.type != 'N/A']

    # sns.set_palette("GnBu_d")

    """ COUNT OF ALL COMMANDS """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=settings.figsize3)
    sns.countplot(data=df_commands, x='participant_id', hue='run_id', ax=ax1)
    sns.countplot(data=df_commands[df_commands.type != 'N/A'], x='participant_id', hue='type', ax=ax2)
    sns.countplot(data=df_commands, x='participant_id', hue='SSD', ax=ax3)
    fig.suptitle('Command Count per Run')
    # ax1.set_title('Subtitle')
    # ax2.set_title('Subtitle')
    ax1.set_xlabel('Particpant ID')
    ax2.set_xlabel('Participant ID')
    ax3.set_xlabel('Participant ID')
    plt.savefig(settings.data_folder + 'figures/command_count.png', bbox_inches='tight')
    plt.close()

    """ COMMAND TYPE PER CONDITION """
    fig, (ax1) = plt.subplots(1, 1, figsize=settings.figsize1)
    sns.countplot(data=df_commands, x='type', hue='SSD', ax=ax1)
    fig.suptitle('Command Count per Run')
    # ax1.set_title('Subtitle')
    # ax2.set_title('Subtitle')
    ax1.set_xlabel('Command type')
    plt.savefig(settings.data_folder + 'figures/command_type.png', bbox_inches='tight')
    plt.close()

    """ DIRECTION AND CONTROL PREFERENCE """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=settings.figsize3)
    data_direction = df_commands[(df_commands.direction != 'N/A')]
    data_direction = data_direction[(data_direction.direction != 'revert')]
    data_direction_spd = data_direction[data_direction.type == 'SPD']
    data_direction_hdg = data_direction[data_direction.type == 'HDG']
    sns.countplot(data=data_direction_spd, x='participant_id', hue='direction', hue_order=['decrease', 'increase'], ax=ax1)
    sns.countplot(data=data_direction_hdg, x='participant_id', hue='direction', hue_order=['left', 'right'], ax=ax2)
    sns.countplot(data=df_commands[df_commands.preference != 'N/A'], x='participant_id', hue='preference', ax=ax3)
    fig.suptitle('Command preferences')
    ax1.set_title('Speed')
    ax2.set_title('Heading')
    ax3.set_title('Control preference')
    ax1.set_xlabel('Particpant ID')
    ax2.set_xlabel('Participant ID')
    plt.savefig(settings.data_folder + 'figures/command_preferences.png', bbox_inches='tight')
    plt.close()

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
    plt.savefig(settings.data_folder + 'figures/command_values.png', bbox_inches='tight')
    plt.close()

    print('test')
    """ FACTOR PLOT COMMAND VALUES """
    hdg_commands = df_commands[df_commands.type == 'HDG']
    pd.options.mode.chained_assignment = None # surprsses the copy warning from following statement:
    hdg_commands.loc[:, 'value'] = hdg_commands.value.apply(lambda x: custom_round(x, base=20))
    pd.options.mode.chained_assignment = 'warn'
    sns.catplot(x='value', col='participant_id', col_wrap=3, data=hdg_commands, kind='count')
    plt.savefig(settings.data_folder + 'figures/facet_command_values.png', bbox_inches='tight')
    plt.close()

    ''' COMMAND TYPE TIMELINE '''
    df_commands_spd = df_commands[df_commands.type == 'SPD']
    g = sns.FacetGrid(df_commands_spd, col="participant_id", col_wrap=3, palette='GnBu_d')
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    plt.suptitle('Speed commands given')
    g.fig.subplots_adjust(top=.9)
    plt.savefig(settings.data_folder + 'figures/commands_over_time_spd.png', bbox_inches='tight')

    df_commands_hdg = df_commands[df_commands.type == 'HDG']
    g = sns.FacetGrid(df_commands_hdg, col="participant_id", col_wrap=3, palette='GnBu_d')
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    plt.suptitle('Heading commands given')
    g.fig.subplots_adjust(top=.9)
    plt.savefig(settings.data_folder + 'figures/commands_over_time_hdg.png', bbox_inches='tight')

    df_commands_dct = df_commands[df_commands.type == 'DCT']
    g = sns.FacetGrid(df_commands_dct, col="participant_id", col_wrap=3, palette='GnBu_d')
    g.map(sns.stripplot, "timestamp", "run_id", "type", jitter=.01)
    plt.suptitle('Direct to commands given')
    g.fig.subplots_adjust(top=.9)
    plt.savefig(settings.data_folder + 'figures/commands_over_time_dct.png', bbox_inches='tight')

    plt.close()


def custom_round(x, base=20):
    """ Rounds number to nearest base """
    return int(base * round(float(x) / base))


def plot_traffic(all_dataframes):

    df_traffic = all_dataframes['traffic'].reset_index()

    """ HEATMAP """
    ''' poging 1 '''
    df_traffic = df_traffic.query("participant_id == 'P1'")
    df_traffic = df_traffic[(df_traffic.x_nm < 35)]
    df_traffic = df_traffic[(df_traffic.x_nm > - 35)]
    df_traffic = df_traffic[(df_traffic.y_nm < 25)]
    df_traffic = df_traffic[(df_traffic.y_nm > - 35)]
    f, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    sns.scatterplot(x=df_traffic.x_nm, y=df_traffic.y_nm, hue=df_traffic.run_id, ax=ax)
    plt.savefig(settings.data_folder + 'figures/scatter.png', bbox_inches='tight')
    plt.show()

    ''' poging 2 '''
    # round x and y to nearest 10. count in buckets then make heatmap.


#
#         ''' AIRCRAFT TYPE '''
#
#         # all_ACIDs = df_commands.ACID.unique().tolist()
#         # if scenario 1:
#         main_flow_ACIDs = ['YG6251', 'XM1337', 'VS4694', 'UM6490', 'UG7514', 'EN5625', 'EF6739', 'AT5763', 'AH2854', 'AN1778']
#         intruding_ACIDs = ['RA4743', 'SG3047', 'SM7071', 'PG4310', 'PA5424', 'RG3628', 'QM2514', 'OS2071', 'NA9895', 'OM3185']
#         # # all aircraft that are not in the main flow are considered intruder aircraft
#         # for ACID in all_ACIDs:
#         #     if ACID not in main_flow_ACIDs:
#         #         intruding_ACIDs.append(ACID)
#
#         df_commands_main = df_commands.loc[df_commands.ACID.isin(main_flow_ACIDs)]
#         df_commands_intruding = df_commands.loc[np.logical_not(df_commands.ACID.isin(main_flow_ACIDs))]
#         num_commands_main = len(df_commands_main)
#         num_commands_intruding = len(df_commands_intruding)
#
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#         sns.countplot(data=df_commands_main, y='ACID', palette='Blues_d', order=main_flow_ACIDs, ax=ax1)
#         ax1.set_title('Main aircraft flow')
#         ax1.set_xlim([0, 10])
#
#         sns.countplot(data=df_commands_intruding, y='ACID', palette='Blues_d', order=intruding_ACIDs, ax=ax2)
#         ax2.set_title('Intruding aircraft flow')
#         ax2.set_xlim([0, 10])
#         # plt.savefig(settings.data_folder + 'figures/ACIDS_run{}.png'.format(i_run), bbox_inches='tight')
#         # if settings.show_plots:
#         #     plt.show()

# def write_to_csv():
    # Create csv file en initialize a writer
    # File names
    # csv_filename = 'data/processed_data.csv'
    #
    # # CSV content
    # csv_header = ["Participant",
    #               "Subject",
    #               "Scenario",
    #               "SectorCoordinates",
    #               "Total commands",
    #               "numOfHDG",
    #               "numOfSPD"]
    #
    # csv_file = open(csv_filename, "w", newline="")
    # csv_writer = csv.writer(csv_file, delimiter=",")
    #
    # # Write header line
    # csv_writer.writerow(csv_header)
    #
    # # csv_writer.writerow([controller.participant,
    # #                      run.subject,
    # #                      run.scenario.file,
    # #                      sector_points,
    # #                      num_commands, num_hdg, num_spd])
    # csv_file.close()


if __name__ == "__main__":
    settings = config.Settings

    # plot settings
    sns.set()
    sns.set_context("notebook")  # smaller: paper

    all_data = pickle.load(open(settings.data_folder + 'all_dataframes.p', "rb"))

    # try:
    #     # participants = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))
    #
    #     print('Data loaded from Pickle')
    # except FileNotFoundError:
    #     print('Pickle all_dataframes.p not found. Please run process_data.py')

    plot_commands(all_data)
    # plot_traffic(all_data)
