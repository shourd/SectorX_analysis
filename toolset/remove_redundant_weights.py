# remove_redundant_weights.py
import numpy as np
import glob
from os import remove

def main():
    output_dir = 'output/weights/'
    target_types = ['direction', 'type', 'value']

    participant_ids = ['all', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ssd_conditions = ['ON', 'OFF', 'BOTH']

    for ssd_condition in ssd_conditions:
        for target_type in target_types:
            for participant_id in participant_ids:
                file_list = glob.glob(output_dir + '{}_{}_*_{}_*.hdf5'.format(
                    target_type,
                    participant_id,
                    ssd_condition))
                file_list.sort(reverse=True)
                try:
                    files_to_be_deleted = file_list[1:]
                    delete_files(files_to_be_deleted)
                except:
                    print('List contains no files.')
    return


def delete_files(file_list):
    if len(file_list) > 1:
        for f in file_list:
            remove(f)


if __name__ == '__main__':
    main()
