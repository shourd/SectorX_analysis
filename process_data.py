from toolset.get_relevant_aircraft import get_relevant_aircraft
import csv
import pickle
from toolset.conflict import get_conflicts
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_dataframes():
    """ IMPORT DATA FROM PICKLE AND SAVE AS PANDAS DATAFRAMES """

    # Load data from pickle
    print("Load serialized data...", end="")
    pickle_file = open(settings.data_folder + settings.serialized_data_filename, "rb")
    participant_list = pickle.load(pickle_file)
    pickle_file.close()
    print("Done!")

    # Get the sector polygon
    sector_points = []
    for sector in participant_list[0].runs[0].scenario.airspace.sectors.sector:
        if sector.type == "sector":
            for point in sector.border_points.point:
                pointX = point.x_nm
                pointY = point.y_nm
                sector_points.append([pointX,pointY])

    # Loop through each participant
    participants = []
    for participant in participant_list:
        runs = []
        # Loop through each run
        for i_run, run in enumerate(participant.runs):

            # print("Analyzing: " + run.participant + ' (run: ' + str(i_run) + ')')
            print('Participant: {} (run: {})'.format(run.participant, i_run+1))
            print('Filename:', run.file_name)

            ''' STATE ANALYSIS '''

            finished_aircraft_list = []  # List of aircraft that reached their goal
            run.conflicts = pd.DataFrame(columns=['timestamp',
                                                  'ACID',
                                                  'Relative ACID',
                                                  'T_CPA',
                                                  'D_CPA',
                                                  'T_LOS',
                                                  'Angle'])

            ''' Loop through each logpoint to create command dataframe '''
            run.traffic = pd.DataFrame()
            score = []
            i_conflict = 0
            for i_logpoint, logpoint in enumerate(run.logpoints):

                score.append(logpoint.score)

                """ ALL AIRCRAFT STATE DATA IS SAVED TO RUN.TRAFFIC """
                # TODO: Misschien dataframe appenden uit de a/c loop halen voor snelheid?
                for aircraft in logpoint.traffic.aircraft:
                    logpoint_data = [i_logpoint, logpoint.timestamp, aircraft.ACID, aircraft.hdg_deg, aircraft.track_cmd, aircraft.spd_kts, aircraft.speed_cmd,
                                     aircraft.x_nm, aircraft.y_nm, aircraft.selected]
                    df = pd.DataFrame(logpoint_data).transpose()
                    run.traffic = run.traffic.append(df)

                if i_logpoint % 50 == 0 and i_logpoint is not 0:
                    print('Analyzed time: {}s'.format(i_logpoint*5))

                """ ALL CONFLICTS ARE SAVED TO RUN.CONFLICTS """

                # Remove aircraft from the logpoint that have reached their
                # destination and have been issued with a TOC command
                relevant_aircraft_list, finished_aircraft_list = \
                    get_relevant_aircraft(
                        logpoint.traffic.aircraft,
                        run.scenario.traffic.aircraft,
                        finished_aircraft_list
                    )

                # Get conflicting aircraft 
                conflict_list = get_conflicts(
                    relevant_aircraft_list,
                    settings.caution_time,
                    settings.warning_time,
                    False, sector_points,
                    settings.distance_to_sector)

                if conflict_list:  # save conflict_list (per timestamp) to conflicts (all timestamps)
                    for conflict in conflict_list:
                        conflict.insert(0, logpoint.timestamp)  # add timestamp to conflict
                        run.conflicts.loc[i_conflict] = conflict
                        i_conflict += 1

            run.conflicts.timestamp = run.conflicts.timestamp.astype(float)

            ''' ALL COMMANDS ARE SAVED IN RUN.COMMAND_LIST '''

            run.traffic.columns = ['logpoint', 'timestamp', 'ACID', 'hdg_deg', 'hdg_comd',
                                   'spd_kts', 'spd_cmd', 'x_nm', 'y_nm', 'selected']

            run.command_list = pd.DataFrame(columns=['timestamp',
                                                     'type',
                                                     'value',
                                                     'ACID',
                                                     'EXQ',
                                                     'timestamp_traffic'])
            run.command_list = run.command_list.astype(dtype=
                                                       {'timestamp': 'float',
                                                        'type': 'object',
                                                        'value': 'int',
                                                        'ACID': 'object',
                                                        'EXQ': 'bool',
                                                        'timestamp_traffic': 'float'})
            for i_command, command in enumerate(run.commands):
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
                    # print('Command type not recognized CMD:', i_command)

                traffic_timestamps = [float(x) for x in run.traffic.timestamp.unique()]  # convert to floats
                command.timestamp_traffic = command.timestamp - 1  # command always taken at previous state
                while command.timestamp_traffic not in traffic_timestamps:
                    command.timestamp_traffic -= 1

                run.command_list.loc[i_command] = [command.timestamp,
                                                   command.type,
                                                   command.value,
                                                   command.ACID,
                                                   command.EXQ,
                                                   command.timestamp_traffic]

            run.command_list = run.command_list[run.command_list.type != 'N/A']  # only take executed commands
            run.command_list.reset_index(drop=True)

            runs.append(run)
        participants.append(runs)

    pickle.dump(participants, open(settings.data_folder + settings.processed_data_filename, "wb"))
    print('Data saved to pickle')
    return participants


def analyse_conflicts(participants):
    """ CONFLICT ANALYSIS """
    for participant in participants:
        for i_run, run in enumerate(participant):

            numerical_columns = ['timestamp', 'Angle', 'T_LOS', 'T_CPA', 'D_CPA']
            run.conflicts[numerical_columns] = run.conflicts[numerical_columns].round(1)  # round all numerical columns

            """ PLOTTING CONFLICTS """
            fig, ax = plt.subplots(figsize=(8, 2))
            sns.stripplot(data=run.conflicts, x='timestamp', ax=ax, jitter=False)
            plt.title('In conflict?')
            ax.set_xlim([0, 1200])
            plt.savefig(settings.data_folder + 'figures/in_conflict_run{}.png'.format(i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            # print('Timesteps with conflict:', len(conflicts))
            # print('Percentage of time in conflict: ', round(len(conflicts) / i_logpoint, 2))


def analyse_commands(participants):
    """ COMMAND / command ANALYSIS """
    for participant in participants:
        for i_run, run in enumerate(participant):

            # CALCULATE NUMBER OF COMMANDS
            num_commands = len(run.command_list)
            num_spd = len(run.command_list[run.command_list.type == 'SPD'])
            num_hdg = len(run.command_list[run.command_list.type == 'HDG'])
            num_dct = len(run.command_list[run.command_list.type == 'DCT'])

            # CALCULATE WHETHER HDG COMMAND IS 'LEFT' OR 'RIGHT'
            # OR SPD COMMAND INCREASE OR DECREASE
            run.command_list['direction'] = 'N/A'
            for i_command, command in run.command_list.iterrows():
                if command.type == 'HDG':
                    # Get HDG (hdg_deg) of respective aircraft (ACID) at the time of the command
                    hdg_current = run.traffic.loc[(run.traffic['timestamp'] == str(command.timestamp_traffic)) & (run.traffic['ACID'] == command.ACID), ['hdg_deg']]

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
                        run.command_list.loc[i_command, 'direction'] = 'right'
                    else:
                        run.command_list.loc[i_command, 'direction'] = 'left'
                elif command.type == 'SPD':
                    if command.value > 250:
                        run.command_list.loc[i_command, 'direction'] = 'increase'
                    elif command.value < 250:
                        run.command_list.loc[i_command, 'direction'] = 'decrease'
                    elif command.value == 250:
                        run.command_list.loc[i_command, 'direction'] = 'revert'

            """ DETERMINE CONTROL PREFERENCE (BEHIND OR IN FRONT) """
            # intruding_ACIDs = run.traffic[(run.traffic.logpoint == 0) & (run.traffic.x_nm > 10)][['ACID']]
            # intruding_ACIDs = list(intruding_ACIDs['ACID'])

            main_flow_ACIDs = run.traffic[(run.traffic.logpoint == 0) & (run.traffic.x_nm < 10)][['ACID']]
            main_flow_ACIDs = list(main_flow_ACIDs['ACID'])

            run.command_list['preference'] = 'N/A'
            for i_command, command in run.command_list.iterrows():
                if command.ACID in main_flow_ACIDs:
                    if command.direction is 'right' or command.direction is 'decrease':
                        run.command_list.loc[i_command, 'preference'] = 'behind'
                    elif command.direction is 'left' or command.direction is 'increase':
                        run.command_list.loc[i_command, 'preference'] = 'infront'
                else:
                    if command.direction is 'left' or command.direction is 'decrease':
                        run.command_list.loc[i_command, 'preference'] = 'behind'
                    elif command.direction is 'right' or command.direction is 'increase':
                        run.command_list.loc[i_command, 'preference'] = 'infront'

            """ HISTOGRAM OF RESOLUTIONS """

            # SPD_mean = run.command_list[run.command_list.type == 'SPD'].mean()
            # SPD_mean = SPD_mean.loc['value']
            # HDG_mean = run.command_list[run.command_list.type == 'HDG'].mean()
            # HDG_mean = HDG_mean.loc['value']
            # print(SPD_mean)
            # print(HDG_mean)

            sns.set()
            spd_commands = run.command_list[run.command_list.type == 'SPD'].value
            spd_commands = pd.to_numeric(spd_commands)
            hdg_commands = run.command_list[run.command_list.type == 'HDG'].value
            hdg_commands_binary = run.command_list[run.command_list.type == 'HDG'].direction
            hdg_commands = pd.to_numeric(hdg_commands)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Histograms of commands')

            sns.distplot(spd_commands, bins=10, kde=False, ax=ax1)
            ax1.set_title('SPD commands')
            ax1.set_xlabel('IAS [kts]')

            sns.distplot(hdg_commands, bins=36, kde=False, ax=ax2)
            ax2.set_title('HDG commands')
            ax2.set_xlabel('HDG [deg]')

            plt.savefig(settings.data_folder + 'figures/command_histogram_run{}.png'.format(i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            ''' LEFT/RIGHT or INCREASE/DECREASE '''
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Relative commands')

            sns.countplot(data=run.command_list[run.command_list.type == 'SPD'],
                          x='direction', order=['decrease', 'revert', 'increase'], ax=ax1)
            ax1.set_title('SPD commands')
            ax1.set_xlabel('IAS')

            sns.countplot(data=run.command_list[run.command_list.type == 'HDG'],
                          x='direction', order=['left', 'right'], ax=ax2)
            ax2.set_title('HDG commands (direction)')
            ax2.set_xlabel('Relative heading')
            plt.savefig(settings.data_folder + 'figures/relative_run{}.png'.format(i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            """ CONTROL PREFERENCE """
            # TODO: Calculate control preferences (and relative commands) in terms of percentages
            fig, ax = plt.subplots()
            sns.countplot(data=run.command_list[run.command_list.preference != 'N/A']
                          , x='preference', ax=ax)
            ax.set_title('Control preference')
            plt.savefig(settings.data_folder + 'figures/preference_run{}.png'.format(i_run), bbox_inches='tight')
            plt.close()

            ''' COMMAND TYPE TIMELINE '''
            fig, ax = plt.subplots()
            sns.stripplot(data=run.command_list, x='timestamp', y='type', jitter=0.01, ax=ax)
            plt.title('Commands given')
            plt.savefig(settings.data_folder + 'figures/commands_run{}.png'.format(i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            ''' AIRCRAFT TYPE '''

            # all_ACIDs = run.command_list.ACID.unique().tolist()
            # if scenario 1:
            main_flow_ACIDs = ['YG6251', 'XM1337', 'VS4694', 'UM6490', 'UG7514', 'EN5625', 'EF6739', 'AT5763', 'AH2854', 'AN1778']
            intruding_ACIDs = ['RA4743', 'SG3047', 'SM7071', 'PG4310', 'PA5424', 'RG3628', 'QM2514', 'OS2071', 'NA9895', 'OM3185']
            # # all aircraft that are not in the main flow are considered intruder aircraft
            # for ACID in all_ACIDs:
            #     if ACID not in main_flow_ACIDs:
            #         intruding_ACIDs.append(ACID)

            run.command_list_main = run.command_list.loc[run.command_list.ACID.isin(main_flow_ACIDs)]
            run.command_list_intruding = run.command_list.loc[np.logical_not(run.command_list.ACID.isin(main_flow_ACIDs))]
            num_commands_main = len(run.command_list_main)
            num_commands_intruding = len(run.command_list_intruding)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            sns.countplot(data=run.command_list_main, y='ACID', palette='Blues_d', order=main_flow_ACIDs, ax=ax1)
            ax1.set_title('Main aircraft flow')
            ax1.set_xlim([0, 10])

            sns.countplot(data=run.command_list_intruding, y='ACID', palette='Blues_d', order=intruding_ACIDs, ax=ax2)
            ax2.set_title('Intruding aircraft flow')
            ax2.set_xlim([0, 10])
            plt.savefig(settings.data_folder + 'figures/ACIDS_run{}.png'.format(i_run), bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            """ CONSISTENCY REPORT """
            print('----------- Consistency Report ------------')
            print('Total commands:', num_commands)
            print('SPD commands:', num_spd)
            print('HDG commands:', num_hdg)
            print('DCT commands:', num_dct)
            if num_spd > num_hdg:
                print('Preferred resolution: SPD ({})'.format(num_spd))

            print('Total commands (Main flow):', num_commands_main)
            print('Total commands (Intruding flow):', num_commands_intruding)


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


if __name__ == "__main__":
    settings = config.Settings
    try:
        participants = pickle.load(open(settings.data_folder + settings.processed_data_filename, "rb"))
        print('Data loaded from Pickle')
    except FileNotFoundError:
        print('Start loading data.')
        participants = create_dataframes()  # contains all data from XML files

    analyse_conflicts(participants)
    analyse_commands(participants)

