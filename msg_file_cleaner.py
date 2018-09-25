""" Removes CLR and PRV commands from the MSG files that interfere with the SSD exporter (MView) """
import os
from config import Settings


def msg_cleaner():
    files = os.listdir(settings.data_folder)
    msg_files = [file for file in files if '.msg' in file]
    for filename in msg_files:
        print('Clearning ', filename)
        msg_file = open(settings.data_folder + filename, "r", newline="")
        command_lines = msg_file.readlines()
        msg_file.close()

        command_lines_clean = []
        for line in command_lines:
            if ':CLR' in line:
                print('Removed CLR command')
            elif ':PRV' in line:
                print('Removed PRV command')
            else:
                command_lines_clean.append(line)

        msg_file_new = open(settings.data_folder + filename[:-4] + '_clean.msg', "w", newline="")
        msg_file_new.writelines(command_lines_clean)
        msg_file_new.close()

        print('File done')
    print('Finished')


if __name__ == "__main__":
    settings = Settings
    msg_cleaner()
