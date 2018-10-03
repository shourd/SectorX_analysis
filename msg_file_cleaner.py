""" Removes CLR and PRV commands from the MSG files that interfere with the SSD exporter (MView) """
import os
from config import Settings


def msg_cleaner():
    files = os.listdir(settings.data_folder_participant)
    msg_files = [file for file in files if '.msg' in file]
    for filename in msg_files:
        print('Cleaning ', filename)
        msg_file = open(settings.data_folder_participant + filename, "r", newline="")
        command_lines = msg_file.readlines()
        msg_file.close()

        command_lines_clean = []
        for line in command_lines:
            if 'HDG' in line or 'SPD' in line or 'DCT' in line or 'rotation' in line:
                command_lines_clean.append(line)

        # msg_file_new = open(settings.data_folder_participant + filename[:-4] + '_clean.msg', "w", newline="")
        msg_file_new = open('data/all/' + filename, "w", newline="")
        msg_file_new.writelines(command_lines_clean)
        msg_file_new.close()

        print('File done')


if __name__ == "__main__":
    settings = Settings
    for i_participant in range(5):
        settings.data_folder_participant = settings.data_folder + '/P{}/'.format(i_participant + 1)
        msg_cleaner()

    print('Finished')
