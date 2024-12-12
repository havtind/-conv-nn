from configparser import ConfigParser


def read_config():
    parser = ConfigParser(converters={'inttuple': parse_int_tuple,
                                      'floattuple': parse_float_tuple,
                                      'strtuple': parse_str_tuple})
    parser.read('config.ini')

    section = 'RUN_OPTIONS'
    visualize_dict = {}
    for option in parser.options(section):
        visualize_dict[option] = parser.getboolean(section, option)

    section = 'DATAGENERATOR'
    data_dict = {'set_size': parser.getint(section, 'set_size'),
                 'split_ratio': parser.getfloattuple(section, 'split_ratio'),
                 'type': parser.get(section, 'type'), 'size': parser.getint(section, 'size'),
                 'flatten': parser.getboolean(section, 'flatten', fallback=False),
                 'variation': parser.getboolean(section, 'variation', fallback=True),
                 'noise_level': parser.getfloat(section, 'noise_level', fallback=0.0),
                 'image_ratio': parser.getfloat(section, 'image/background_ratio', fallback=0.7)}

    section = 'NETWORK'
    network_dict = {'epochs': parser.getint(section, 'epochs', fallback=1),
                    'minibatch_size': parser.getint(section, 'minibatch_size'),
                    'learning_rate': parser.getfloat(section, 'learning_rate'),
                    'loss_function': parser.get(section, 'loss_function'),
                    'softmax': parser.getboolean(section, 'softmax'),
                    'smooth_labels': parser.getboolean(section, 'smooth_labels')}
    if data_dict['type'] == '1D':
        network_dict['input_size'] = data_dict['size']
    elif data_dict['type'] == '2D' and data_dict['flatten']:
        network_dict['input_size'] = data_dict['size']*data_dict['size']
    elif data_dict['type'] == '2D':
        network_dict['input_size'] = (data_dict['size'],data_dict['size'])

    section_index = 1
    layers_dict = {}
    while True:
        section = f'{section_index}'
        if not parser.has_section(f'{section_index}'):
            break
        if not parser.has_option(section, 'type'):
            section_index += 1
            continue
        layers_dict[section] = {}
        layers_dict[section]['type'] = parser.get(section, 'type')
        layers_dict[section]['activation'] = parser.get(section, 'activation', fallback='linear')
        layers_dict[section]['show_hinton'] = parser.getboolean(section, 'show_hinton', fallback=False)
        layers_dict[section]['verbose'] = parser.getboolean(section, 'verbose', fallback=False)
        if parser.get(section, 'weight_initialization') == 'glorot':
            layers_dict[section]['weight_init'] = 'glorot'
        else:
            layers_dict[section]['weight_init'] = parser.getfloattuple(section, 'weight_initialization')
        if layers_dict[section]['type'] == 'conv':
            layers_dict[section]['number_of_filters'] = parser.getint(section, 'number_of_filters')
            if len(parser.get(section, 'filter_size')) > 3:
                layers_dict[section]['filter_size'] = parser.getinttuple(section, 'filter_size')
            else:
                layers_dict[section]['filter_size'] = parser.getint(section, 'filter_size')
            if len(parser.get(section, 'mode')) > 7:
                layers_dict[section]['mode'] = parser.getstrtuple(section, 'mode')
            else:
                layers_dict[section]['mode'] = str(parser.get(section, 'mode'))
            if len(parser.get(section, 'stride')) > 4:
                layers_dict[section]['stride'] = parser.getinttuple(section, 'stride')
            else:
                layers_dict[section]['stride'] = parser.getint(section, 'stride')
        elif layers_dict[section]['type'] == 'dense':
            layers_dict[section]['neurons'] = parser.getint(section, 'neurons')
        section_index += 1

    return visualize_dict, data_dict, network_dict, layers_dict



# Specialized converters

def parse_int_tuple(tup_str):
    return tuple(int(k.strip()) for k in tup_str[1:-1].split(','))


def parse_float_tuple(tup_str):
    return tuple(float(k.strip()) for k in tup_str[1:-1].split(','))


def parse_str_tuple(tup_str):
    return tuple(str(k.strip()) for k in tup_str[1:-1].split(','))
