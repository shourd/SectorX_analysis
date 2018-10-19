import os
import xml.etree.ElementTree as ET
import pickle
import re
from toolset.data_objects import Participant, Run, Command
from config import settings


def serialize_data():
    """ Convert XML files to python Classes/objects """
    print("Start serialization...")

    # Controller list
    participant_list = []

    # Obtain list of xml files and remove .xml extension
    files = os.listdir(settings.data_folder)
    xml_files = [file[:-4] for file in files if '.xml' in file]
    print('Number of runs:', len(xml_files))
    xml_files = sorted(xml_files, key=lambda x: int(x[-1]))  # sorts the list based on run number !important
    xml_files = [file+'.xml' for file in xml_files]
    for i_file, file in enumerate(xml_files):
        i_run = int(file[-5])
        participant_names = [participant.name for participant in participant_list]
        participant_name = file.split("_", 1)[0]
        if participant_name not in participant_names:
            participant_list.append(Participant(participant_name))
            print('New participant created')

        print("Process " + file + "...")

        # obtain participant number (index) in list
        for index, participant in enumerate(participant_list):
            if participant.name == participant_name:
                i_participant = index

        # parse XML
        root_element = ET.parse(settings.data_folder + file).getroot()
        # Deserialize XML into objects
        run = Run(root_element, file[:-4])

        """ CONTINUE WITH MSG FILE """
        file = file[:-4] + '.msg' # change .xml filename to .msg filename
        # Read the commands from the message file and skip the header
        msg_file = open(settings.data_folder + file, "r", newline="")
        subject = re.findall("subject:(.*),", msg_file.readline())[0]
        command_lines = msg_file.readlines()
        msg_file.close()

        # Deserialize MSG into objects
        run.commands = []
        for command in command_lines:
            run.commands.append(Command(command.strip()))

        participant_list[i_participant].runs.append(run)

    # Serialize the controller list
    pickle_file = open(settings.data_folder + settings.serialized_data_filename, "wb")
    pickle.dump(participant_list, pickle_file)
    pickle_file.close()
    print("Saved serialized data to " + settings.serialized_data_filename)

    return participant_list


if __name__ == "__main__":
    serialize_data()
    print('Finished')

