import pickle
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import settings
from toolset.conflict import get_conflicts
from toolset.get_relevant_aircraft import get_relevant_aircraft

def create_dataframes(participant_list=None):
    """
    :return: Three dataframes:
    all_dataframes.traffic
    all_dataframes.commands
    all_dataframes.conflicts
    IMPORT DATA AND CONVERT TO PANDAS DATAFRAMES
    """
    if participant_list is None:
        participant_list = load_from_pickle(settings.serialized_data_filename)

    sector_points = obtain_sector_points(participant_list)

    experiment_setup = initialize_experiment_setup()

    df_commands = initialize_command_dataframe(only_columns=True)
    df_traffic = initialize_traffic_dataframe()

    copx_list = obtain_copx_list(participant_list)

    for i_participant, participant in enumerate(participant_list):

        for i_run, run in enumerate(participant.runs):

            participant_id = int(participant_list[i_participant].name[1:])
            run_id = 'R{}'.format(i_run + 1)
            run.SSD = experiment_setup.loc[run_id, participant_id].SSD
            run.scenario_id = experiment_setup.loc[run_id, participant_id].Scenario

            print('Participant: {} (run: {}/{})'.format(run.participant, i_run + 1, len(participant.runs)))
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
                    aircraft_temp_df = copx_list.loc[(copx_list.Scenario == run.scenario_id) &
                                                       (copx_list.ACID == aircraft.ACID)]
                    aircraft.copx_x_nm = aircraft_temp_df.copx_x_nm[0]
                    aircraft.copx_y_nm = aircraft_temp_df.copx_y_nm[0]

                    temp_traffic_df.loc[i_aircraft] = [participant_id, run.SSD, run_id, i_logpoint, logpoint.timestamp,
                                                       aircraft.ACID, aircraft.conflict, aircraft.controlled,
                                                       aircraft.hdg_deg, aircraft.selected, aircraft.spd_kts,
                                                       aircraft.x_nm, aircraft.y_nm, aircraft.copx_x_nm, aircraft.copx_y_nm]

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

    """ Create indeces for dataframes """
    participant_ids = list(df_traffic.participant_id.unique())
    run_ids = list(df_traffic.run_id.unique())

    df_commands = df_commands.reset_index()
    df_commands = df_commands.set_index(['participant_id', 'run_id', 'index'])
    df_commands.index.names = ['participant_id', 'run_id', 'i_command']
    df_commands = df_commands[df_commands.type != 'N/A']
    df_commands['direction'] = 'N/A'
    df_commands['preference'] = 'N/A'
    df_commands['hdg_rel'] = 'N/A'

    df_traffic = df_traffic.reset_index()
    df_traffic = df_traffic.set_index(['participant_id', 'run_id', 'index'])
    df_traffic.index.names = ['participant_id', 'run_id', 'i_aircraft']
    df_traffic.timestamp = df_traffic.timestamp.astype('float')

    all_dataframes = {
        "traffic": df_traffic.sort_index(),
        "commands": df_commands.sort_index(),
        "conflicts": run.conflicts,
        "participants": participant_ids,
        "runs": run_ids,
        "sector_points": sector_points}

    pickle.dump(all_dataframes, open(settings.data_folder + 'all_dataframes_1.p', "wb"))
    print('Processed data saved to pickle')

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


def analyse_commands(all_dataframes=None):
    """ COMMAND ANALYSIS
    :type all_dataframes: dict
    """

    if all_dataframes is None:
        all_dataframes = load_from_pickle('all_dataframes.p')

    df_traffic = all_dataframes['traffic']
    df_commands = all_dataframes['commands']

    df_traffic = add_copx_heading_column(df_traffic)

    for participant_id in all_dataframes['participants']:
        for run_id in all_dataframes['runs']:
            df_traffic_run = df_traffic.loc[(participant_id, run_id)]
            df_commands_run = df_commands.loc[(participant_id, run_id)]

            directions, hdg_rel_list = determine_directional_values(df_traffic_run, df_commands_run)
            df_commands.loc[(participant_id, run_id), 'direction'] = directions
            df_commands.loc[(participant_id, run_id), 'hdg_rel'] = hdg_rel_list

            preferences = determine_control_preference(df_traffic_run, df_commands_run)
            df_commands.loc[(participant_id, run_id), 'preference'] = preferences

    df_commands = df_commands[df_commands.type != 'N/A']
    all_dataframes['commands'] = df_commands
    all_dataframes['traffic'] = df_traffic

    pickle.dump(all_dataframes, open(settings.data_folder + 'all_dataframes_2.p', "wb"))
    print('Command data saved to pickle')

    return all_dataframes


def add_copx_heading_column(df_traffic):
    pos_points_array = np.array([df_traffic.x_nm, df_traffic.y_nm])
    copx_points_array = np.array([df_traffic.copx_x_nm, df_traffic.copx_y_nm])
    delta_points_array = copx_points_array - pos_points_array

    heading_array_rad = np.arctan2(delta_points_array[0,:], delta_points_array[1,:])
    heading_array_rad += (heading_array_rad < [0]*len(heading_array_rad)) * 2 * np.pi
    heading_array_deg = heading_array_rad * 360 / (2 * np.pi)

    df_traffic['copx_hdg'] = heading_array_deg.astype(int)

    return df_traffic


def determine_directional_values(df_traffic, df_commands):
    """
    CALCULATE WHETHER HDG COMMAND IS 'LEFT' OR 'RIGHT'
    OR SPD COMMAND INCREASE OR DECREASE
    """

    directions = ['N/A'] * len(df_commands)
    relative_hdg_list = ['N/A'] * len(df_commands)

    for i_command in range(len(df_commands)):
        command = df_commands.iloc[i_command]
        try:
            if command.type == 'HDG' or command.type == 'DCT':
                # if command.type == 'DCT':
                # Get HDG (hdg_deg) of respective aircraft (ACID) at the time of the command
                traffic_logpoint = df_traffic.loc[
                    (df_traffic['timestamp'] == command.timestamp_traffic) & (df_traffic['ACID'] == command.ACID)]

                hdg_current = traffic_logpoint.hdg_deg.iloc[0]

                if command.type == 'DCT':
                    hdg_resolution = traffic_logpoint.copx_hdg.iloc[0]
                else:
                    hdg_resolution = command.value

                hdg_relative = hdg_resolution - hdg_current

                # make sure hdg_rel is always between -180 and 180
                if hdg_relative > 180:
                    hdg_relative -= 360
                elif hdg_relative < -180:
                    hdg_relative += 360

                if command.type == 'DCT' and abs(hdg_relative) < 0:
                    relative_hdg_list[i_command] = 'N/A'  # if DCT bearing smaller than 10 deg, do not include. TURNED OFF
                else:
                    relative_hdg_list[i_command] = int(hdg_relative)

                # add direction value to Commands table
                if hdg_relative > 0:
                    directions[i_command] = 'right'
                else:
                    directions[i_command] = 'left'
            elif command.type == 'SPD':
                spd_current = df_traffic.loc[
                    (df_traffic['timestamp'] == command.timestamp_traffic) & (df_traffic['ACID'] == command.ACID), [
                        'spd_kts']]
                hdg_current = spd_current.iloc[0][0]  # take value only
                spd_resolution = command.value
                spd_relative = spd_resolution - hdg_current

                if command.value > 250:
                    directions[i_command] = 'increase'
                elif command.value < 250:
                    directions[i_command] = 'decrease'
                elif command.value == 250:
                    directions[i_command] = 'revert'


        except IndexError:
            print(df_commands.to_string())
            print('Index Error!')

    return directions, relative_hdg_list


def determine_control_preference(df_traffic, df_commands):
    """ DETERMINE CONTROL PREFERENCE (BEHIND OR IN FRONT) """

    preferences = ['N/A'] * len(df_commands)

    main_flow_ACIDs = df_traffic[(df_traffic.i_logpoint == 0) & (df_traffic.x_nm < 10)][['ACID']]
    main_flow_ACIDs = list(main_flow_ACIDs['ACID'])
    commands_received_list = []  # list of ACIDS who have received a command

    for i_command in range(len(df_commands)):
        command = df_commands.iloc[i_command]
        if command.ACID not in commands_received_list:
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

            commands_received_list.append(command.ACID)
    return preferences


def initialize_experiment_setup():
    """ Create experiment set-up dataframe """
    arrays = [['R1', 'R1', 'R2', 'R2', 'R3', 'R3', 'R4', 'R4'],
              ['Scenario', 'SSD', 'Scenario', 'SSD', 'Scenario', 'SSD', 'Scenario', 'SSD']]

    experiment_setup = pd.DataFrame(columns=np.arange(1,13,1), index=arrays)
    experiment_setup.loc[(['R1', 'R3'], 'Scenario'), :] = 'S1'
    experiment_setup.loc[(['R2', 'R4'], 'Scenario'), :] = 'S2'

    experiment_setup.loc[(['R1', 'R2'], 'SSD'), [1, 2, 3, 7, 8, 9]] = 'OFF'
    experiment_setup.loc[(['R3', 'R4'], 'SSD'), [4, 5, 6, 10, 11, 12]] = 'OFF'

    experiment_setup.loc[(['R3', 'R4'], 'SSD'), [1, 2, 3, 7, 8, 9]] = 'ON'
    experiment_setup.loc[(['R1', 'R2'], 'SSD'), [4, 5, 6, 10, 11, 12]] = 'ON'

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

    return command_list


def initialize_traffic_dataframe():
    """ INITIALIZE TRAFFIC DATAFRAME """
    columns = ['participant_id', 'SSD', 'run_id',
               'i_logpoint', 'timestamp',
               'ACID', 'conflict', 'controlled', 'hdg_deg',
               'selected', 'spd_kts', 'x_nm', 'y_nm', 'copx_x_nm', 'copx_y_nm']

    command_list = pd.DataFrame(columns=columns)

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


def obtain_copx_list(participant_list):
    copx_df = pd.DataFrame()
    for i_run in [0,1]:
        for aircraft in participant_list[0].runs[i_run].scenario.traffic.aircraft:
            if aircraft.COPX == 'MIFA': aircraft.copx_y_nm = 22.0  # hardcoded due to error. Remove.
            df = pd.DataFrame({'Scenario': 'S{}'.format(i_run+1),
                               'ACID': aircraft.ACID,
                               'COPX': aircraft.COPX,
                               'copx_x_nm': aircraft.COPX_x_nm,
                               'copx_y_nm': aircraft.COPX_y_nm},
                              index=[0])
            if copx_df.empty:
                copx_df = df
            else:
                copx_df = copx_df.append(df)

    return copx_df

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
    elif command.TOC is True:  # transfer of control
        command.type = 'TOC'
        command.value = None
    else:
        command.type = 'N/A'
        command.value = None

    return command.type


def load_from_pickle(filename):
    print("Loading data from pickle...", end="")
    pickle_file = open(settings.data_folder + filename, "rb")
    loaded_data = pickle.load(pickle_file)
    pickle_file.close()
    print("Done!")

    return loaded_data


if __name__ == "__main__":

    # try:
    #     # participants = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))
    #     all_data = pickle.load(open(settings.data_folder + 'all_dataframes_1.p', "rb"))
    #     print('Data loaded from Pickle')
    # except FileNotFoundError:
    print('Start loading data.')
    all_data = create_dataframes()  # generates all_dataframes_1.p
    print('Dataframes created and saved to Pickle')

    all_data = analyse_commands(all_data)  # generates all_dataframes_2.p
    print('Commands analyzed and saved to Pickle')
    # analyse_conflicts(all_data)

