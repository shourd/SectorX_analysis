"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

from toolset.get_relevant_aircraft import getRelevantAircraft
import csv
import sys
import pickle
import math
import numpy
from toolset.conflict import getConflicts

# Flush prints
sys.stdout.flush()

# Directory locations
data_folder = '/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data/'
processed_data_folder = '/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data/'

# File names
csv_filename = 'processed_data.csv'

# CSV content
csv_header = ["Participant",
              "Subject",
              "Scenario",
              "SectorCoordinates",
              "Total actions",
              "numOfHDG",
              "numOfSPD"]

# Controller list
controller_list = list()


def main():

    # Notify user
    print("Load serialized data...", end="", flush=True)

    # Deserialize data
    pickle_file = open(data_folder + "serializedData.p", "rb")
    controller_list = pickle.load(pickle_file)
    pickle_file.close()

    # Notify user
    print("Done!", flush=True)

    # Create csv file en initialize a writer
    csv_file = open(processed_data_folder + csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file, delimiter=",")

    # Write header line
    csv_writer.writerow(csv_header)

    # Get the sector polygon
    sector_points = []
    for sector in controller_list[0].experiment[0].record.scenario.airspace.sectors.sector:
        if sector.type == "sector":
            for point in sector.border_points.point:
                pointX = point.x_nm
                pointY = point.y_nm
                sector_points.append([pointX,pointY])

    # Loop through each controller
    for controller in controller_list:

        # Loop through each experiment
        for experiment in controller.experiment:

            # Notify user
            print("Analyzing " + experiment.record.participant + " " + experiment.subject + " (" + experiment.recordXML + ")...")

            # List of aircraft that reached their goal
            finished_aircraft_list = []

            # Loop through each logpoint
            for iLogpoint, logpoint in enumerate(experiment.record.logpoints):

                # Remove aircraft from the logpoint that have reached their
                # destination and have been issued with a TOC command
                relevant_aircraft_list, finished_aircraft = \
                    getRelevantAircraft(
                        logpoint.traffic.aircraft,
                        experiment.record.scenario.traffic.aircraft,
                        finished_aircraft_list
                    )

                finished_aircraft_list = finished_aircraft

                # Get conflicting aircraft 
                conflictList = getConflicts(relevant_aircraft_list, 300, 160, False, sector_points, 50)

                # print(conflictList)

                # Loop through the conflicts
                # for conflict in conflictList:

                #    # Loop through each aircraft
                #    iAircraft: int = None
                #    aircraft: logpointAircraft = None
                #    for iAircraft, aircraft in enumerate(relevant_aircraft_list):

            # Number of EFL, HDG, SPD
            num_actions, num_hdg, num_spd = [0] * 3

            # Loop through each command in the experiment
            for command in experiment.commands:

                num_actions += 1

                # If HDG command is given
                if command.HDG != None:
                    num_hdg += 1

                # If SPD command is given
                if command.SPD != None:
                    num_spd += 1

            # Write row
            csv_writer.writerow([controller.participant,
                                experiment.record.subject,
                                experiment.record.scenario.file,
                                sector_points,
                                num_actions, num_hdg, num_spd])

            print('Total actions: ', num_actions)
            print('SPD actions: ', num_spd)
            print('HDG actions: ', num_hdg)

    csv_file.close()


if __name__ == "__main__":
    """ If stand-alone """
    main()
