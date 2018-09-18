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
    print("Load serialized data...", end="", flush=True)
    pickle_file = open("data/serialized_data.p", "rb")
    controller_list = pickle.load(pickle_file)
    pickle_file.close()
    print("Done!", flush=True)

    # Get the sector polygon
    sector_points = []
    for sector in controller_list[0].experiment[0].record.scenario.airspace.sectors.sector:
        if sector.type == "sector":
            for point in sector.border_points.point:
                pointX = point.x_nm
                pointY = point.y_nm
                sector_points.append([pointX,pointY])

    # Loop through each controller
    participants = []
    for controller in controller_list:
        runs = []
        # Loop through each run
        for run in controller.experiment:

            # Notify user
            print("Analyzing " + run.record.participant + " " + run.subject + " (" + run.recordXML + ")...")

            ''' STATE ANALYSIS '''

            finished_aircraft_list = []  # List of aircraft that reached their goal
            run.conflicts = pd.DataFrame(columns=['time',
                                                  'ACID',
                                                  'Relative ACID',
                                                  'T_CPA',
                                                  'D_CPA',
                                                  'T_LOS',
                                                  'Angle'])

            ''' Loop through each logpoint to create command dataframe '''
            run.traffic = pd.DataFrame()
            i_conflict = 0
            for i_logpoint, logpoint in enumerate(run.record.logpoints):

                """ ALL AIRCRAFT STATE DATA IS SAVED TO RUN.TRAFFIC """

                for aircraft in logpoint.traffic.aircraft:
                    logpoint_data = [i_logpoint, aircraft.ACID, aircraft.hdg_deg, aircraft.track_cmd, aircraft.spd_kts, aircraft.speed_cmd,
                                     aircraft.x_nm, aircraft.y_nm, aircraft.selected]
                    df = pd.DataFrame(logpoint_data).transpose()
                    run.traffic = run.traffic.append(df)

                if i_logpoint % 200 == 0:
                    print('Analyzed time: {}s'.format(i_logpoint))

                """ ALL CONFLICTS ARE SAVED TO RUN.CONFLICTS """

                # Remove aircraft from the logpoint that have reached their
                # destination and have been issued with a TOC command
                relevant_aircraft_list, finished_aircraft_list = \
                    get_relevant_aircraft(
                        logpoint.traffic.aircraft,
                        run.record.scenario.traffic.aircraft,
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
                        conflict.insert(0, i_logpoint)  # add timestamp to conflict
                        run.conflicts.loc[i_conflict] = conflict
                        i_conflict += 1

            ''' ALL COMMANDS ARE SAVED IN RUN.COMMAND_LIST '''

            run.traffic.columns = ['time', 'ACID', 'hdg_deg', 'hdg_comd',
                                   'spd_kts', 'spd_cmd', 'x_nm', 'y_nm','selected']
            run.command_list = pd.DataFrame(columns=['time', 'type', 'value', 'ACID', 'EXQ'])
            run.command_list = run.command_list.astype(dtype=
                                               {'time': 'int',
                                                'type': 'object',
                                                'value': 'int',
                                                'ACID': 'object',
                                                'EXQ': 'bool'})
            for i_command, command in enumerate(run.commands):
                # Define command type
                if command.HDG is not None:
                    command.type = 'HDG'
                    command.value = command.HDG
                if command.SPD is not None:
                    command.type = 'SPD'
                    command.value = command.SPD
                if command.DCT is True:  # direct to exit waypoint
                    command.type = 'DCT'
                    command.value = None

                run.command_list.loc[i_command] = [command.timestamp,
                                               command.type,
                                               command.value,
                                               command.ACID,
                                               command.EXQ]

            run.command_list = run.command_list[run.command_list.EXQ == True]  # only take executed values
            # print(run.command_list)

            runs.append(run)
        participants.append(runs)

    pickle.dump(participants, open("data/all_data.pickle", "wb"))
    print('Data saved to pickle')
    return participants


def analyse_conflicts(participants):
    """ CONFLICT ANALYSIS """
    for participant in participants:
        for run in participant:

            float_columns = ['Angle', 'T_LOS', 'T_CPA', 'D_CPA']
            run.conflicts[float_columns] = run.conflicts[float_columns].round(1)  #round all numerical columns

            # print(run.conflicts)
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.stripplot(data=run.conflicts, x='time', ax=ax, jitter=False)
            plt.title('In conflict?')
            plt.savefig('figures/in_conflict.png', bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            # print('Timesteps with conflict:', len(conflicts))
            # print('Percentage of time in conflict: ', round(len(conflicts) / i_logpoint, 2))


def analyse_actions(participants):
    """ COMMAND / ACTION ANALYSIS """
    for participant in participants:
        for run in participant:

            # CALCULATE NUMBER OF ACTIONS
            num_actions = len(run.command_list)
            num_spd = len(run.command_list[run.command_list.type == 'SPD'])
            num_hdg = len(run.command_list[run.command_list.type == 'HDG'])
            num_dct = len(run.command_list[run.command_list.type == 'DCT'])

            # CALCULATE WHETHER HDG COMMAND IS 'LEFT' OR 'RIGHT'
            # OR SPD COMMAND INCREASE OR DECREASE
            run.command_list['direction'] = 'N/A'
            for index, row in run.command_list.iterrows():
                if row.type == 'HDG':
                    # Get HDG (hdg_deg) of respective aircraft (ACID) at the time of the command
                    hdg_current = run.traffic.loc[(run.traffic['time'] == row.time) & (run.traffic['ACID'] == row.ACID), ['hdg_deg']]
                    hdg_current = hdg_current.iloc[0][0]  # take value only
                    hdg_resolution = row.value

                    hdg_relative = hdg_resolution - hdg_current
                    # make sure hdg_rel is always between -180 and 180
                    if hdg_relative > 180:
                        hdg_relative -= 360
                    elif hdg_relative < -180:
                        hdg_relative += 360

                    # add direction value to Commands table
                    if hdg_relative > 0:
                        run.command_list.loc[index, 'direction'] = 'right'
                    else:
                        run.command_list.loc[index, 'direction'] = 'left'
                elif row.type == 'SPD':
                    if row.value > 250:
                        run.command_list.loc[index, 'direction'] = 'increase'
                    elif row.value < 250:
                        run.command_list.loc[index, 'direction'] = 'decrease'
                    elif row.value == 250:
                        run.command_list.loc[index, 'direction'] = 'revert'

            """ HISTOGRAM OF RESOLUTIONS """

            SPD_mean = run.command_list[run.command_list.type == 'SPD'].mean()
            SPD_mean = SPD_mean.loc['value']
            HDG_mean = run.command_list[run.command_list.type == 'HDG'].mean()
            HDG_mean = HDG_mean.loc['value']
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

            plt.savefig('figures/action_histogram.png', bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            ''' LEFT/RIGHT or INCREASE/DECREASE '''
            # TODO: MAKE BARPLOT with increase decrease left right.
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
            # if settings.show_plots:
            plt.show()

            ''' COMMAND TYPE TIMELINE '''
            fig, ax = plt.subplots()
            sns.stripplot(data=run.command_list, x='time', y='type', jitter=0.01, ax=ax)
            plt.title('Commands given')
            plt.savefig('figures/commands.png', bbox_inches='tight')
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
            num_actions_main = len(run.command_list_main)
            num_actions_intruding = len(run.command_list_intruding)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            sns.countplot(data=run.command_list_main, y='ACID', palette='Blues_d', order=main_flow_ACIDs, ax=ax1)
            ax1.set_title('Main aircraft flow')
            ax1.set_xlim([0, 10])

            sns.countplot(data=run.command_list_intruding, y='ACID', palette='Blues_d', order=intruding_ACIDs, ax=ax2)
            ax2.set_title('Intruding aircraft flow')
            ax2.set_xlim([0, 10])
            plt.savefig('figures/ACIDS.png', bbox_inches='tight')
            if settings.show_plots:
                plt.show()

            """ CONSISTENCY REPORT """
            print('----------- Consistency Report ------------')
            print('Total actions:', num_actions)
            print('SPD actions:', num_spd)
            print('HDG actions:', num_hdg)
            print('DCT actions:', num_dct)
            if num_spd > num_hdg:
                print('Preferred resolution: SPD ({})'.format(num_spd))

            print('Total actions (Main flow):', num_actions_main)
            print('Total actions (Intruding flow):', num_actions_intruding)


def write_to_csv():
    # Create csv file en initialize a writer
    # File names
    csv_filename = 'data/processed_data.csv'

    # CSV content
    csv_header = ["Participant",
                  "Subject",
                  "Scenario",
                  "SectorCoordinates",
                  "Total actions",
                  "numOfHDG",
                  "numOfSPD"]

    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=",")

    # Write header line
    csv_writer.writerow(csv_header)

    # csv_writer.writerow([controller.participant,
    #                      run.record.subject,
    #                      run.record.scenario.file,
    #                      sector_points,
    #                      num_actions, num_hdg, num_spd])
    csv_file.close()


if __name__ == "__main__":
    """ If stand-alone """
    settings = config.Settings
    try:
        participants = pickle.load(open("data/all_data.pickle", "rb"))
        print('Data loaded from Pickle')
    except FileNotFoundError:
        print('Start loading data.')
        participants = create_dataframes()  # contains all data from XML files

    analyse_conflicts(participants)
    analyse_actions(participants)

