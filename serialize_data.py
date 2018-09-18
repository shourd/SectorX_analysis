import os
import xml.etree.ElementTree as ET
import pickle
import sys
import re
from data_objects import Participant, Run, Run, Command
from config import Settings


def serialize_data():
    """ Convert XML files to python Classes/objects """
    print("Start serialization...")

    settings = Settings

    # Controller list
    participant_list = []
    participant_names = []

    # Obtain list of xml files
    files = os.listdir(settings.data_folder)
    xml_files = [file for file in files if '.xml' in file]
    for i_file, file in enumerate(xml_files):
        print("Process " + file + "...")

        participant_name = file.split("_", 1)[0]
        if participant_name not in participant_names:
            participant_names.append(participant_name)
            participant_list.append(Participant(participant_name))
        else:
            print('Participant already exists')

        # obtain participant number (index) in list
        for index, participant in enumerate(participant_list):
            if participant.name == participant_name:
                i_participant = index

        # parse XML
        root_element = ET.parse(settings.data_folder + file).getroot()
        # Deserialize XML into objects
        run = Run(root_element, file[:-4])

        """ CONTINUE WITH MSG FILE """
        # change .xml filename to .msg filename
        file = file[:-4] + '.msg'
        print("Process " + file + "...")
        # Read the commands from the message file and skip the header
        msg_file = open(settings.data_folder + file, "r", newline="")
        subject = re.findall("subject:(.*),", msg_file.readline())[0]
        command_lines = msg_file.readlines()[1:]
        msg_file.close()

        # Deserialize MSG into objects
        run.commands = []
        for command in command_lines:
            run.commands.append(Command(command.strip()))

        participant_list[i_participant].runs.append(run)

    print("Saved serialized data to " + settings.serialized_data_filename)

    # Serialize the controller list
    pickle_file = open(settings.data_folder + settings.serialized_data_filename, "wb")
    pickle.dump(participant_list, pickle_file)
    pickle_file.close()


if __name__ == "__main__":
    serialize_data()

