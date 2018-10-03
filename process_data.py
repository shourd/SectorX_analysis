from toolset.get_relevant_aircraft import get_relevant_aircraft
import csv
import pickle
from toolset.conflict import get_conflicts
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os import mkdir
sns.set()
sns.set_context("notebook")  # smaller: paper


def create_dataframes():
    """
    :return: Three dataframes:
    all_dataframes.traffic
    all_dataframes.commands
    all_dataframes.conflicts
    IMPORT DATA AND CONVERT TO PANDAS DATAFRAMES
    """

    participant_list = load_from_pickle()
    sector_points = obtain_sector_points(participant_list)

    experiment_setup = initialize_experiment_setup()

    df_commands = initialize_command_dataframe(only_columns=True)
    df_traffic = initialize_traffic_dataframe()

    participants = []
    for i_participant, participant in enumerate(participant_list):
        runs = []
        for i_run, run in enumerate(participant.runs):

            participant_id = participant_list[i_participant].name
            run_id = 'R{}'.format(i_run + 1)
            run.SSD = experiment_setup.loc[run_id, participant_id].SSD
            run.scenario_id = experiment_setup.loc[run_id, participant_id].Scenario

            print('Participant: {} (run: {})'.format(run.participant, i_run + 1))
            print('Filename:', run.file_name)
            print('SSD: {}'.format(run.SSD))

            ''' STATE ANALYSIS '''

            finished_aircraft_list = []  # List of aircraft that reached their goal
            run.conflicts = pd.DataFrame(columns=['timestamp',
                                                  'ACID',
                                                  'Relative ACID',
                                                  'T_CPA',
                                                  'D_CPA',
                                                  'T_LOS',
                                                  'Angle'])

            ''' Loop through each logpoint to fill dataframes '''
            score = []
            i_conflict = 0

            temp_traffic_df = initialize_traffic_dataframe()
            for i_logpoint, logpoint in enumerate(run.logpoints):

                score.append(logpoint.score)

                """ CREATING TRAFFIC DATAFRAME """
                for i_aircraft, aircraft in enumerate(logpoint.traffic.aircraft):
                    temp_traffic_df.loc[i_aircraft] = [participant_id, run.SSD, run_id, i_logpoint, logpoint.timestamp,
                                                       aircraft.ACID, aircraft.conflict, aircraft.controlled,
                                                       aircraft.hdg_deg, aircraft.selected, aircraft.spd_kts,
                                                       aircraft.x_nm, aircraft.y_nm]

                df_traffic = df_traffic.append(temp_traffic_df)

                if i_logpoint % 40 == 0 and i_logpoint is not 0:
                    print('Analyzed time: {}s'.format(i_logpoint*5))

                """ CREATING CONFLICT DATAFRAME """
                # Remove aircraft from the logpoint that have reached their
                # destination and have been issued with a TOC command
                relevant_aircraft_list, finished_aircraft_list = \
                    get_relevant_aircraft(
                        logpoint.traffic.aircraft,
                        run.scenario.traffic.aircraft,
                        finished_aircraft_list
                    )

                conflict_list = get_conflicts(relevant_aircraft_list, sector_points, settings)  # Get conflicting aircraft

                if conflict_list:  # save conflict_list (per timestamp) to conflicts (all timestamps)
                    for conflict in conflict_list:
                        conflict.insert(0, logpoint.timestamp)  # add timestamp to conflict
                        run.conflicts.loc[i_conflict] = conflict
                        i_conflict += 1

            run.conflicts.timestamp = run.conflicts.timestamp.astype(float)

            ''' ALL COMMANDS ARE SAVED IN df_commands '''
            temp_command_list = initialize_command_dataframe(only_columns=True)

            # all unique timestamp values per run and convert to float
            traffic_timestamps = df_traffic[(df_traffic.participant_id == participant_id) & (df_traffic.run_id == run_id)].timestamp.unique()
            traffic_timestamps = [float(timestamp) for timestamp in traffic_timestamps]

            for i_command, command in enumerate(run.commands):

                command.type = determine_command_type(command)

                command.timestamp_traffic = command.timestamp - 1  # command always taken at previous state
                while command.timestamp_traffic not in traffic_timestamps:
                    command.timestamp_traffic -= 1

                temp_command_list.loc[i_command] = [participant_id,
                                                    run.SSD,
                                                    run_id,
                                                    run.scenario_id,
                                                    command.timestamp,
                                                    command.type,
                                                    command.value,
                                                    command.ACID,
                                                    command.timestamp_traffic]

            df_commands = pd.concat([df_commands, temp_command_list])

            # runs.append(run)
        # participants.append(runs)

    """ Create indeces for dataframes """
    participant_ids = list(df_traffic.participant_id.unique())
    run_ids = list(df_traffic.run_id.unique())

    df_commands = df_commands.reset_index()
    df_commands = df_commands.set_index(['participant_id', 'run_id', 'index'])
    df_commands.index.names = ['participant_id', 'run_id', 'i_command']
    df_commands['direction'] = 'N/A'
    df_commands['preference'] = 'N/A'

    df_traffic = df_traffic.reset_index()
    df_traffic = df_traffic.set_index(['participant_id', 'run_id', 'index'])
    df_traffic.index.names = ['participant_id', 'run_id', 'i_aircraft']

    all_dataframes = {
        "traffic": df_traffic.sort_index(),
        "commands": df_commands.sort_index(),
        "conflicts": run.conflicts,
        "participants": participant_ids,
        "runs": run_ids,
        "sector_points": sector_points}

    # pickle.dump(participants, open(settings.data_folder + settings.processed_data_filename, "wb"))
    pickle.dump(all_dataframes, open(settings.data_folder + 'all_dataframes.p', "wb"))
    print('Data saved to pickle')

    return all_dataframes


def analyse_conflicts(participants):
    """ CONFLICT ANALYSIS """
    for participant in participants:

        participant_id = participant[0].participant
        try:
            mkdir(settings.data_folder + 'figures/{}'.format(participant_id))
        except FileExistsError:
            pass

        for i_run, run in enumerate(participant):

            numerical_columns = ['timestamp', 'Angle', 'T_LOS', 'T_CPA', 'D_CPA']
            run.conflicts[numerical_columns] = run.conflicts[numerical_columns].round(1)  # round all numerical columns

            """ PLOTTING CONFLICTS """
            fig, ax = plt.subplots(figsize=(8, 2))
            sns.stripplot(data=run.conflicts, x='timestamp', ax=ax, jitter=False)
            plt.title('In conflict?')
            ax.set_xlim([0, 1200])
            plt.savefig(settings.data_folder + 'figures/{}/in_conflict_run{}.png'.format(participant_id, i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()
            plt.close()

            # print('Timesteps with conflict:', len(conflicts))
            # print('Percentage of time in conflict: ', round(len(conflicts) / i_logpoint, 2))


def analyse_commands(all_dataframes):
    """ COMMAND ANALYSIS """

    # TODO: rerun pickel then remove sort_index here.
    df_commands = all_dataframes['commands'].sort_index()
    df_commands = df_commands[df_commands.type != 'N/A']
    df_traffic = all_dataframes['traffic'].sort_index()

    for participant_id in all_dataframes['participants']:
        for run_id in all_dataframes['runs']:

            df_traffic_run = df_traffic.loc[(participant_id, run_id)]
            df_commands_run = df_commands.loc[(participant_id, run_id)]
            directions = determine_directional_values(df_traffic_run, df_commands_run)
            df_commands.loc[(participant_id, run_id), 'direction'] = directions

            preferences = determine_control_preference(df_traffic_run, df_commands_run)
            df_commands.loc[(participant_id, run_id), 'preference'] = preferences

    df_commands = df_commands[df_commands.type != 'N/A']
    all_dataframes['commands'] = df_commands
    all_dataframes['traffic'] = df_traffic

    pickle.dump(all_dataframes, open(settings.data_folder + 'all_dataframes.p', "wb"))
    print('Data saved to pickle')

    return all_dataframes


def plot_commands(all_dataframes):
    df_commands = all_dataframes['commands'].reset_index()
    # df_traffic = all_dataframes['traffic'].reset_index()

    df_commands = df_commands[df_commands.type != 'N/A']

    # sns.set_palette("GnBu_d")

    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)

    """ COUNT OF ALL COMMANDS """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize2)
    sns.countplot(data=df_commands, x='participant_id', hue='run_id', ax=ax1)
    sns.countplot(data=df_commands[df_commands.type != 'N/A'], x='participant_id', hue='type', ax=ax2)
    fig.suptitle('Title')
    # ax1.set_title('Subtitle')
    # ax2.set_title('Subtitle')
    ax1.set_xlabel('Particpant ID')
    ax2.set_xlabel('Participant ID')
    plt.savefig(settings.data_folder + 'figures/command_count.png', bbox_inches='tight')
    plt.close()

    """ DIRECTION AND CONTROL PREFERENCE """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize3)
    data_direction = df_commands[(df_commands.direction != 'N/A')]
    data_direction = data_direction[(data_direction.direction != 'revert')]
    data_direction_spd = data_direction[data_direction.type == 'SPD']
    data_direction_hdg = data_direction[data_direction.type == 'HDG']
    sns.countplot(data=data_direction_spd, x='participant_id', hue='direction', hue_order=['decrease', 'increase'], ax=ax1)
    sns.countplot(data=data_direction_hdg, x='participant_id', hue='direction', hue_order=['left', 'right'], ax=ax2)
    sns.countplot(data=df_commands[df_commands.preference != 'N/A'], x='participant_id', hue='preference', ax=ax3)
    fig.suptitle('Title')
    ax1.set_title('Speed')
    ax2.set_title('Heading')
    ax3.set_title('Control preference')
    ax1.set_xlabel('Particpant ID')
    ax2.set_xlabel('Participant ID')
    plt.savefig(settings.data_folder + 'figures/command_preferences.png', bbox_inches='tight')
    plt.close()

    """ COMMAND VALUES"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize2)
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

    """ FACTOR PLOT COMMAND VALUES """
    hdg_commands = df_commands[df_commands.type == 'HDG']

    hdg_commands.value = hdg_commands.value.apply(lambda x: custom_round(x, base=20))
    sns.catplot(x='value', col='participant_id', col_wrap=3, data=hdg_commands, kind='count')
    plt.savefig(settings.data_folder + 'figures/facet_command_values.png', bbox_inches='tight')

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

    plt.show()
    plt.close()

def custom_round(x, base=20):
        return int(base * round(float(x) / base))

def plot_traffic(all_dataframes):
    figsize1 = (5, 5)
    figsize2 = (10, 5)
    figsize3 = (15, 5)

    df_traffic = all_dataframes['traffic'].reset_index()

    """ HEATMAP """
    ''' poging 1 '''
    df_traffic = df_traffic.query("participant_id == 'P1'")
    df_traffic = df_traffic[(df_traffic.x_nm < 35)]
    df_traffic = df_traffic[(df_traffic.x_nm > - 35)]
    df_traffic = df_traffic[(df_traffic.y_nm < 25)]
    df_traffic = df_traffic[(df_traffic.y_nm > - 35)]
    f, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax = sns.scatterplot(x=df_traffic.x_nm, y=df_traffic.y_nm, hue=df_traffic.run_id)
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


def determine_directional_values(df_traffic, df_commands):
    """
    CALCULATE WHETHER HDG COMMAND IS 'LEFT' OR 'RIGHT'
    OR SPD COMMAND INCREASE OR DECREASE
    """

    directions = ['N/A'] * len(df_commands)

    for i_command, command in df_commands.iterrows():
        if command.type == 'HDG':
            # Get HDG (hdg_deg) of respective aircraft (ACID) at the time of the command
            hdg_current = df_traffic.loc[
                (df_traffic['timestamp'] == str(command.timestamp_traffic)) & (df_traffic['ACID'] == command.ACID), [
                    'hdg_deg']]
            hdg_current = hdg_current.iloc[0][0]  # take value only
            hdg_resolution = command.value
            hdg_relative = hdg_resolution - hdg_current
            # make sure hdg_rel is always between -180 and 180
            if hdg_relative > 180:
                hdg_relative -= 360
            elif hdg_relative < -180:
                hdg_relative += 360
            # add direction value to Commands table
            if hdg_relative > 0:
                directions[i_command] = 'right'
            else:
                directions[i_command] = 'left'
        elif command.type == 'SPD':
            if command.value > 250:
                directions[i_command] = 'increase'
            elif command.value < 250:
                directions[i_command] = 'decrease'
            elif command.value == 250:
                directions[i_command] = 'revert'

    return directions


def determine_control_preference(df_traffic, df_commands):
    """ DETERMINE CONTROL PREFERENCE (BEHIND OR IN FRONT) """

    preferences = ['N/A'] * len(df_commands)

    main_flow_ACIDs = df_traffic[(df_traffic.i_logpoint == 0) & (df_traffic.x_nm < 10)][['ACID']]
    main_flow_ACIDs = list(main_flow_ACIDs['ACID'])

    for i_command, command in df_commands.iterrows():
        if command.ACID in main_flow_ACIDs:
            if command.direction is 'right' or command.direction is 'decrease':
                preferences[i_command] = 'behind'
            elif command.direction is 'left' or command.direction is 'increase':
                preferences[i_command] = 'infront'
        else:
            if command.direction is 'left' or command.direction is 'decrease':
                preferences[i_command] = 'behind'
            elif command.direction is 'right' or command.direction is 'increase':
                preferences[i_command] = 'infront'

    return preferences


def write_to_csv():
    # Create csv file en initialize a writer
    # File names
    csv_filename = 'data/processed_data.csv'

    # CSV content
    csv_header = ["Participant",
                  "Subject",
                  "Scenario",
                  "SectorCoordinates",
                  "Total commands",
                  "numOfHDG",
                  "numOfSPD"]

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=",")

    # Write header line
    csv_writer.writerow(csv_header)

    # csv_writer.writerow([controller.participant,
    #                      run.subject,
    #                      run.scenario.file,
    #                      sector_points,
    #                      num_commands, num_hdg, num_spd])
    csv_file.close()


def initialize_experiment_setup():
    """ Create experiment set-up dataframe """
    arrays = [['R1', 'R1', 'R2', 'R2', 'R3', 'R3', 'R4', 'R4'],
              ['Scenario', 'SSD', 'Scenario', 'SSD', 'Scenario', 'SSD', 'Scenario', 'SSD']]

    experiment_setup = pd.DataFrame(columns=settings.columns, index=arrays)
    experiment_setup.loc[(['R1', 'R3'], 'Scenario'), :] = 'S1'
    experiment_setup.loc[(['R2', 'R4'], 'Scenario'), :] = 'S2'

    experiment_setup.loc[(['R1', 'R2'], 'SSD'), ['P1', 'P2', 'P3', 'P7', 'P8', 'P9']] = 'OFF'
    experiment_setup.loc[(['R3', 'R4'], 'SSD'), ['P4', 'P5', 'P6', 'P10', 'P11', 'P12']] = 'OFF'

    experiment_setup.loc[(['R3', 'R4'], 'SSD'), ['P1', 'P2', 'P3', 'P7', 'P8', 'P9']] = 'ON'
    experiment_setup.loc[(['R1', 'R2'], 'SSD'), ['P4', 'P5', 'P6', 'P10', 'P11', 'P12']] = 'ON'

    return experiment_setup


def initialize_command_dataframe(only_columns=False):
    """ INITIALIZE COMMAND DATAFRAME """
    columns = ['participant_id', 'SSD', 'run_id', 'scenario', 'timestamp', 'type', 'value', 'ACID', 'timestamp_traffic']
    if only_columns:
        command_list = pd.DataFrame(columns=columns)
    else:
        iterables = [settings.columns, ['ON', 'OFF']]
        index = pd.MultiIndex.from_product(iterables, names=['participant', 'SSD'])
        command_list = pd.DataFrame(columns=columns, index=index)

    # command_list = command_list.astype(dtype=
    #                                    {'run': 'int',
    #                                     'scenario': 'object',
    #                                     'timestamp': 'float',
    #                                     'type': 'object',
    #                                     'value': 'int',
    #                                     'ACID': 'object',
    #                                     'timestamp_traffic': 'float'})

    return command_list


def obtain_sector_points(participant_list):
    sector_points = []
    for sector in participant_list[0].runs[0].scenario.airspace.sectors.sector:
        if sector.type == "sector":
            for point in sector.border_points.point:
                pointX = point.x_nm
                pointY = point.y_nm
                sector_points.append([pointX, pointY])
    return sector_points


def determine_command_type(command):
    # Define command type
    if command.HDG is not None:
        command.type = 'HDG'
        command.value = command.HDG
    elif command.SPD is not None:
        command.type = 'SPD'
        command.value = command.SPD
    elif command.DCT is True:  # direct to exit way-point
        command.type = 'DCT'
        command.value = None
    else:
        command.type = 'N/A'
        command.value = None

    return command.type


def initialize_traffic_dataframe():
    """ INITIALIZE TRAFFIC DATAFRAME """
    columns = ['participant_id', 'SSD', 'run_id',
               'i_logpoint', 'timestamp',
               'ACID', 'conflict', 'controlled', 'hdg_deg',
               'selected', 'spd_kts', 'x_nm', 'y_nm']
    command_list = pd.DataFrame(columns=columns)

    return command_list


def load_from_pickle():
    print("Loading data from pickle...", end="")
    pickle_file = open(settings.data_folder + settings.serialized_data_filename, "rb")
    participant_list = pickle.load(pickle_file)
    pickle_file.close()
    print("Done!")

    return participant_list


if __name__ == "__main__":
    settings = config.Settings
    settings.data_folder = settings.data_folder + '/all/'

    try:
        # participants = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))
        all_data = pickle.load(open(settings.data_folder + 'all_dataframes.p', "rb"))
        print('Data loaded from Pickle')
    except FileNotFoundError:
        print('Start loading data.')
        all_data = create_dataframes()  # contains all data from XML files
        all_data = analyse_commands(all_data)

    # analyse_conflicts(participants)
    # plot_commands(all_data)
    plot_traffic(all_data)


