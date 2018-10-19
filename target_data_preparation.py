from config import settings

def make_categorical_list(command_data, target_type):
    """
    :param command_data:
    :param target_type: type can be 1. command type 2. command direction (left/right) or 3. command geometry
    :return: target_list
    """
    target_list = []

    """ COMMAND TYPE """
    if target_type is 'command_type':
        for command in list(command_data.TYPE):
            if command == 'HDG': res = 0
            elif command == 'SPD': res = 1
            elif command == 'DCT': res = 2
            elif command == 'TOC': res = 3
            else:
                print('ERROR: Command target_type not recognized')
                break
            target_list.append(res)

    elif target_type is 'direction':
        settings.class_names = ['Left', 'Right']
        for command in list(command_data.direction):
            if command == 'left': res = 0
            elif command == 'right': res = 1
            else:
                print('ERROR: Command target_type not recognized')
                break
            target_list.append(res)

        number_left = target_list.count(0)
        number_right = target_list.count(1)
        total = number_left + number_right
        if total == 0:
            print('ERROR: No data was selected!')
        print('Left: {} ({}%)'.format(number_left, round(100*number_left/total),0))
        print('Right: {} ({}%)'.format(number_right, round(100*number_right/total),0))

    elif target_type is 'geometry':
        settings.class_names = ['In front', 'Behind']
        for command in list(command_data.preference):
            if command == 'infront':
                res = 0
            elif command == 'behind':
                res = 1
            else:
                print('ERROR: Command target_type not recognized')
                break
            target_list.append(res)

        number_infront = target_list.count(0)
        number_behind = target_list.count(1)
        total = number_infront + number_behind
        print('In front: {} ({}%)'.format(number_infront, round(100 * number_infront / total), 0))
        print('Behind: {} ({}%)'.format(number_behind, round(100 * number_behind / total), 0))

    return target_list