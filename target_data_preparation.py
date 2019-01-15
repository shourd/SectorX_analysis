from config import settings


def make_categorical_list(command_data, target_type):
    """
    :param command_data:
    :param target_type: type can be:
        1. type (spd, hdg, dct)
        2. direction (left/right)
        3. geometry
        4. value (only hdg)
    :return: target_list
    """
    target_list = []

    """ COMMAND TYPE """
    if target_type is 'type':
        settings.class_names = ['HDG', 'SPD', 'DCT']
        for command in list(command_data.type):
            if command == 'HDG': res = 0
            elif command == 'SPD': res = 1
            elif command == 'DCT': res = 2
            elif command == 'TOC': res = 3
            else:
                print('ERROR: Command target_type not recognized')
                break
            target_list.append(res)

        number_HDG = target_list.count(0)
        number_SPD = target_list.count(1)
        number_DCT = target_list.count(2)
        total = number_HDG + number_SPD + number_DCT
        if total == 0:
            print('ERROR: No data was selected!')
        print('HDG: {} ({}%)'.format(number_HDG, round(100 * number_HDG / total),0))
        print('SPD: {} ({}%)'.format(number_SPD, round(100 * number_SPD / total),0))
        print('DCT: {} ({}%)'.format(number_DCT, round(100 * number_DCT / total), 0))

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

    elif target_type is 'value':

        print('Samples in dataset:', len(command_data))
        # sns.set()
        # sns.distplot(list(abs(command_data.hdg_rel)), bins=18)
        # plt.savefig('{}/{}_dist.png'.format(settings.output_dir, settings.iteration_name), bbox_inches='tight')
        # plt.close()

        settings.class_names = ['0-10 deg', '10-45 deg', '> 45 deg']
        settings.num_classes = 3
        line1 = 10
        line2 = 45

        for command in list(command_data.hdg_rel):
            if command <= line1:
                res = 0
            elif line1 < command <= line2:
                res = 1
            elif command > line2:
                res = 2
            else:
                res = 3
                print('RELATIVE HEADING ERROR!')

            target_list.append(res)

        print('0-10 deg: {} samples. 10-45 deg: {} samples. > 45 deg: {} samples'.format(target_list.count(0), target_list.count(1), target_list.count(2)))
        print('Total HDG commands: {}, divided into {} classes.'.format(len(target_list), settings.num_classes))

    return target_list
