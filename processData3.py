"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

from toolset.getRelevantAircraft import getRelevantAircraft
import csv
import sys
import pickle
import math
import numpy
from toolset.conflict import getConflicts

# Flush prints
sys.stdout.flush()

# Directory locations
dataFolder = '/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data/'
processedDataFolder = '/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data/'

# File names
csvFileName = "processedData3.csv"

# CSV content
csvHeader = ["Participant",
             "Subject",
             "Scenario",
             "SectorCoordinates",
             "Total actions", "numOfHDG", "numOfSPD"]

# Controller list
controllerList = list()


def main():

    # Notify user
    print("Load serialized data...", end="", flush=True)

    # Deserialize data
    pickleFile = open(dataFolder + "serializedData.p", "rb")
    controllerList = pickle.load(pickleFile)
    pickleFile.close()

    # Notify user
    print("Done!", flush = True)

    # Create csv file en initialize a writer
    csvFile = open(processedDataFolder + csvFileName, "w", newline="")
    csvWriter = csv.writer(csvFile, delimiter=",")

    # Write header line
    csvWriter.writerow(csvHeader)

    # Get the sector polygon
    sectorPoints = []
    for sector in controllerList[0].experiment[0].record.scenario.airspace.sectors.sector:
        if sector.type == "sector":
            for point in sector.border_points.point:
                pointX = point.x_nm
                pointY = point.y_nm
                sectorPoints.append([pointX,pointY])

    # Loop through each controller
    for controller in controllerList:

        # Loop through each experiment
        for experiment in controller.experiment:

            # Notify user
            print("Analyzing " + experiment.record.participant + " " + experiment.subject + " (" + experiment.recordXML + ")...")

            # List of aircraft that reached their goal
            finishedAircraftList = []

            # Loop through each logpoint
            for iLogpoint, logpoint in enumerate(experiment.record.logpoints):

                # Remove aircraft from the logpoint that have reached their destination and have been issued with a TOC command
                relevantAircraftList, finishedAircraft = getRelevantAircraft(logpoint.traffic.aircraft, experiment.record.scenario.traffic.aircraft, finishedAircraftList)
                finishedAircraftList = finishedAircraft

                # Get conflicting aircraft 
                conflictList = getConflicts(relevantAircraftList, 300, 160, False, sectorPoints, 50)

                # print(conflictList)

                ## Loop through the conflicts
                #for conflict in conflictList:

                #    # Loop through each aircraft
                #    iAircraft: int = None
                #    aircraft: logpointAircraft = None
                #    for iAircraft, aircraft in enumerate(relevantAircraftList):

            # Number of EFL, HDG, SPD
            numActions, numOfHDG, numOfSPD = [0] * 3

            # Loop through each command in the experiment
            for command in experiment.commands:

                numActions += 1

                # If HDG command is given
                if command.HDG != None:
                    numOfHDG += 1

                # If SPD command is given
                if command.SPD != None:
                    numOfSPD += 1

            # Write row
            csvWriter.writerow([controller.participant,
                                experiment.record.subject,
                                experiment.record.scenario.file,
                                sectorPoints,
                                numActions, numOfHDG, numOfSPD])

            print('Total actions: ', numActions)
            print('SPD actions: ', numOfSPD)
            print('HDG actions: ', numOfHDG)

    csvFile.close()


if __name__ == "__main__":
    main()
