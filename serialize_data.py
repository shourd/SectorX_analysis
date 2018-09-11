"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

# This file serializes the individual data files

import os
import xml.etree.ElementTree as ET
import pickle
import sys
import re
from data_objects import Controller, Experiment, Record, Command

# Directory loactions
dataFolder = "/Users/sjoerdvanrooijen/PycharmProjects/SectorX/data/"

# Serialized data filename
serializedDataFilename = "serialized_data.p"

# Controller list
controllerList = list()


def main():

    # Notify user
    print("Start serialization...", flush=True)

    # Loop through each data file in the data folder
    for dataFile in os.listdir(dataFolder):

        # Record and commands variable
        record = None
        commands = None

        # If the file is an xml or msg file
        if (dataFile.endswith(".xml")) or (dataFile.endswith(".msg")):

            # Notify user
            print("Process " + dataFile + "...", end="", flush=True)

            # If controller already exist
            if 'newController' not in locals() or newController.participant != dataFile.split(".", 1)[0]:
                # Create a new controller
                newController = Controller(dataFile.split(".", 1)[0], list())
                controllerList.append(newController)
                index = -1
            else:
                print('Participant already exists')

            # If the file is an xml file
            if dataFile.endswith(".xml"):

                # Get the root element
                rootElement = ET.parse(dataFolder + dataFile).getroot()

                # Deserialize XML into objects
                record = Record(rootElement, dataFile.split(".", 1)[0])

                # If the experiment is yet present
                # if any(record.subject in experiment.subject for experiment in controllerList[index].experiment):
                #     experimentIndex = next(i for i,x in enumerate(controllerList[index].experiment) if x.subject == record.subject)
                #     controllerList[index].experiment[experimentIndex].record = record
                #     controllerList[index].experiment[experimentIndex].recordXML = dataFile
                # else:
                #     newExperiment = Experiment(record.subject, record, None, dataFile, None)
                #     controllerList[index].experiment.append(newExperiment)
                index = 0
                experimentIndex = 0
                controllerList[index].experiment[experimentIndex].record = record
                controllerList[index].experiment[experimentIndex].recordXML = dataFile

            # If the file is an msg file
            if dataFile.endswith(".msg"):

                # Read the commands from the message file and skip the header
                msgFile = open(dataFolder + dataFile, "r", newline="")
                subject = re.findall("subject:(.*),", msgFile.readline())[0]
                commandLines = msgFile.readlines()[1:]
                msgFile.close()

                # Deserialize MSG into objects
                commands = list()
                for commandLine in commandLines:
                    commands.append(Command(commandLine.strip()))

                # If the experiment is yet present
                if any(subject in experiment.subject for experiment in controllerList[index].experiment):
                    experimentIndex = next(i for i, x in enumerate(controllerList[index].experiment) if x.subject == subject)
                    controllerList[index].experiment[experimentIndex].commands = commands
                    controllerList[index].experiment[experimentIndex].commandsMSG = dataFile
                else:
                    newExperiment = Experiment(subject, None, commands, None, dataFile)
                    controllerList[index].experiment.append(newExperiment)

            # Notify user
            print("Done!", flush=True)

    # Notify user
    print("Serialize processed data to " + serializedDataFilename + "...", end="", flush = True)

    # Serialize the controller list
    pickleFile = open(dataFolder + serializedDataFilename, "wb")
    pickle.dump(controllerList, pickleFile)
    pickleFile.close()

    # Notify user
    print("Done!", flush = True)
    print("Finished!", flush = True)


if __name__ == "__main__":
    main()

