# This file takes the "raw" data from /home/ytsboe/data/boats/raw_data
# and puts it in a computer readable format, so we can analyse the data
# (for cleaning later)
#
# Data format currently:
#
# builder_name:             beneteau usa
# type_name:                beneteau first 22
# category:                 zeiljacht
# build_material:           polyester
# build_year:               1982.0
# asking_price_euros:       3700.0
# length_over_all_meters:   6.6
# width_meters:             2.5
# draft_meters:             0.65
# displaces_kgs:            1100.0
# ballast_kgs:              None
# steering:                 tiller
# engine_name:              None
# engine_build_year:        None
# standing_height_meters:   None
# url:                      https://www.yachtfocus.com/gebruikte-boten-te-koop/details/XXX/88261/XXX
#
# Data format computer readable:
#
# For the input data a matrix like so:
# [[row1], [row2], ...]
#
# where each columns has floats with the following meaning:
# [length_over_all_meters,
#  build_year,
#  width_meters,
#  draft_meters,
#  displaces_kgs,
#  ballast_kgs
#  builder1,                               # Note: the builder1 will be 0 or 1
#  builder2,                                       depending on whether the boat
#  ...                                             has that builder or not.
#  ]
#
# If a value is not given, the average value is put in place.
# The length of each row is len(FEATURE_NAMES) + len(BUILDERS_COUNT_DICT)
#
# And for the asking price a matrix (with 1 row)
# of which each column corresponds the 1 boat.
# Of course column i of the data matrix corresponds to the boat
# of price of column i in the asking price matrix.
#
# Note, we will apply cuts!

import os  
import numpy as np

DATA_DIR_IN = 'boats/raw_data'
DATA_DIR_OUT = 'boats/computer_readable_data'
MIN_BUILDERS_COUNT = 25
FEATURE_NAMES = ('length_over_all_meters',
                 'width_meters',
                 'build_year',
                 'draft_meters',
                 'displaces_kgs',
                 'ballast_kgs',
                 'engine_build_year')

FEATURE_NAMES = ('length_over_all_meters',
                 'build_year')


PRINT_DEBUG = True


def myprint(* args):

    if not PRINT_DEBUG:
        return

    s = ''

    for arg in args:
        s += arg

    print(s)


def get_file_full_paths():

    """
    This function returns the list of files in the
    directory DATA_DIR_IN.
    """

    file_full_path_list = list()

    for file_name in os.listdir(DATA_DIR_IN):

        file = '{}/{}'.format(DATA_DIR_IN, file_name)
        if not os.path.isfile(file):
            print('Skipping {}'.format(file))
            continue

        file_full_path_list.append(file)

    return file_full_path_list


def get_builders_count_dict():

    """
    This function fetches the builders from the raw data.
    It writes them to a dictionary, with key the builder name
    and value the count.
    Any boats occuring less than MIN_BUILDERS_COUNT are lumped
    together with the unkown boats under name 'None'.
    
    That dictionary will be used for the neural net and linear regression
    elsewhere in the code.
    """

    builders_dict_raw = dict()

    for file in get_file_full_paths():

        with open(file) as f:

            for line in f:

                if 'builder_name' in line:
                    builder = line.split(':')[1].split()[0]

                    if builder:
                        try:
                            builders_dict_raw[builder] += 1
                        except KeyError:
                            builders_dict_raw[builder] = 1

    # Now fetch only the most frequest boats
    # and lump the rest ('None') together
    total_boats_count = 0
    low_freq_count = 0
    builders_dict = dict()
    for builder, count in builders_dict_raw.items():

        total_boats_count += count

        if count < MIN_BUILDERS_COUNT:
            low_freq_count += count
        else:
            try:
                builders_dict[builder] += count
            except KeyError:
                builders_dict[builder] = count

    try:
        builders_dict['None'] += low_freq_count
    except KeyError:
        builders_dict['None'] = low_freq_count

    print('total_boats_count = {}'.format(total_boats_count))

    # Now create final dict with proper (large'ish) counts
    total_boats_count = 0
    for builder, count in builders_dict.items():

        total_boats_count += count

        print('{}: {}'.format(count, builder))

    print('total_boats_count = {}'.format(total_boats_count))

    return builders_dict


BUILDERS_COUNT_DICT = get_builders_count_dict()
print('BUILDERS_COUNT_DICT = {}'.format(BUILDERS_COUNT_DICT))

BUILDER_NAMES = [k for k in BUILDERS_COUNT_DICT.keys()]


def get_value_from_line(line):

    val = line.split(':')[1]
    if 'non' in val.lower():
        val = None
    else:
        val = float(val)

    return val


def get_boat_data_cr_dicts_from_files():

    """
    This function fetches the boat data in a dictionary for each boat.
    It returns a list of boat data dicts.
    """

    dict_list = list()

    for file in get_file_full_paths():

        with open(file) as f:

            # print('Starting file: {}'.format(file))

            boat_data = dict()
            boat_data['file_path'] = f.name

            for line in f:

                for key in FEATURE_NAMES:

                    if key in line:

                        # Don't confuse engine_build_year with build_year
                        if key == 'build_year' and 'engine' in line:
                            continue
                        
                        boat_data[key] = get_value_from_line(line)

                if 'asking_price_euros' in line:
                    boat_data['asking_price_euros'] = get_value_from_line(line)

                if 'builder_name' in line:
                    boat_data['builder_name'] = line.split(':')[1].split()[0]

            # print('Got boat data: {}'.format(boat_data))
            dict_list.append(boat_data)

    return dict_list

    
def remove_outliers(data_dicts):

    """
    This function applies the current cuts.
    """

    cut_data_dicts = list()

    for data_dict in data_dicts:

        passes_cuts = True

        for key, value in data_dict.items():

            # No worries if no value, we substitute averages later
            if not value:
                continue

            if key == 'asking_price_euros':
                if value < 1000. or value > 500000.:
                    passes_cuts = False
                    break

            if key == 'length_over_all_meters':
                if value < 3. or value > 30.:
                    passes_cuts = False
                    break

            if key == 'width_meters':
                if value < 1.5 or value > 5.5:
                    passes_cuts = False
                    break

            if key == 'build_year':
                if value < 1960. or value > 2016.:
                    passes_cuts = False
                    break

            if key == 'engine_build_year':
                if value < 1975. or value > 2016.:
                    passes_cuts = False
                    break

            if key == 'draft_meters':
                if value < 0.1 or value > 3.3:
                    passes_cuts = False
                    break

            if key == 'displaces_kgs':
                if value < 100. or value > 25000.:
                    passes_cuts = False
                    break

            if key == 'ballast_kgs':
                if value < 100. or value > 4000.:
                    passes_cuts = False
                    break

        if passes_cuts:
            cut_data_dicts.append(data_dict)
        else:
            print('Cutting {}'.format(data_dict))
            
    return cut_data_dicts


def transform_data(data_dicts):

    """
    This function transforms some input data.

    It takes the natural logarithm of the asking
    prices.
    """

    for data_dict in data_dicts:

        val = data_dict['asking_price_euros']
        val = np.log(val)
        data_dict['asking_price_euros'] = val

    return data_dicts


def substitute_averages(data_dicts):

    """
    This function creates the rows of the data matrix
    as defined above.

    It substitutes averages for numerical values, which are missing.
    """

    # Collect sums and counts, preparing to calculate averages
    sum_dict = dict()
    count_dict = dict()

    for data_dict in data_dicts:

        for key, value in data_dict.items():

            if key in FEATURE_NAMES or key == 'asking_price_euros':

                if value:
                    try:
                        sum_dict[key] += value
                        count_dict[key] += 1
                    except:
                        sum_dict[key] = value
                        count_dict[key] = 1

    # Calculate averages
    av_dict = dict()
    for key in sum_dict.keys():

        den = float(count_dict[key])
        if den == 0.:
            print('ERROR: Zero denominator for key {}'.format(key))
            print('       Not providing average.')

        av_dict[key] = float(sum_dict[key]) / den
        # print('key: {} sum: {} count: {} av: {}'.format(
        #     key,
        #     sum_dict[key],
        #     count_dict[key],
        #     av_dict[key]))

    print('\nAverages:')
    for k, v in av_dict.items():
        print('  {}: {}'.format(k, v))

    # Substitute None's with averages
    subsituted_dicts = list()
    for data_dict in data_dicts:
        subst_dict = dict()
        subst_dict['file_path'] = data_dict['file_path']
        for key, value in data_dict.items():
            if key in FEATURE_NAMES or key == 'asking_price_euros':
                if value:
                    subst_dict[key] = value
                else:
                    subst_dict[key] = av_dict[key]

            elif key == 'builder_name':

                keys_list = [k for k in BUILDERS_COUNT_DICT.keys()]
                sparse_list = np.zeros(len(keys_list))
                if value in BUILDERS_COUNT_DICT.keys():
                    sparse_list[keys_list.index(value)] = 1.
                else:
                    sparse_list[keys_list.index('None')] = 1.
                subst_dict[key] = sparse_list

        subsituted_dicts.append(subst_dict)

    # print('Found {} completed dicts:'.format(len(subsituted_dicts)))
    # for c_dict in subsituted_dicts:
    #     for k,v in c_dict.items():
    #         print('  {}: {}'.format(k, v))
    #     print('----')

    return subsituted_dicts, av_dict


def get_boat_data_cr(data_dicts):

    """
    This function extends the numerical data with the
    sparse array for the builders.
    It returns a matrix, of which the
    colums are the input vectors for training.
    """

    boat_count = len(data_dicts)
    col_count = len(FEATURE_NAMES) + len(BUILDERS_COUNT_DICT.keys())

    print('boat_count = {}'.format(boat_count))
    print('col_count  = {}'.format(col_count))

    input_data = np.zeros(boat_count * col_count)
    target_data = np.zeros(boat_count)
    file_paths = list()

    for i in range(len(data_dicts)):

        col0 = i * col_count

        data_dict = data_dicts[i]
        file_paths.append(data_dict['file_path'])

        # print('data_dict:')
        # for key in FEATURE_NAMES:
        #     print('  {}: {}'.format(key, data_dict[key]))
        # print('  builder_name: {}'.format(data_dict['builder_name']))

        for key in FEATURE_NAMES:
            input_data[col0 + FEATURE_NAMES.index(key)] = data_dict[key]

            # print('(key={}) input_data[{}] = {}'.format(
            #     key,
            #     col0 + FEATURE_NAMES.index(key),
            #     data_dict[key]))

        col1 = col0 + len(FEATURE_NAMES)
        col2 = col1 + len(BUILDERS_COUNT_DICT.keys())

        input_data[col1:col2] = data_dict['builder_name']
        target_data[i] = data_dict['asking_price_euros']

    input_data = input_data.reshape((boat_count, col_count))
    target_data = target_data.reshape((1, boat_count))

    return np.transpose(input_data), target_data, file_paths


def map_to_min1_plus1(val, min, max):

    """
    This function maps the value "val",
    assumed to be on [min, max], onto [-1, 1].
    """

    return -1. + 2. * (val - min) / (max - min)


def normalise_to_min1_plus1(data_dicts):

    """
    This function normalises all the data features,
    including the target, but excluding the builder info,
    to [-1, 1].

    This is a useful range for the neural nets, becaus the
    transfer functions we use for map the input data to that
    range so we expect no large weights to compensate.
    """

    float_info = np.finfo('float64')

    normed_data_dicts = list()

    min_dict = dict()
    max_dict = dict()

    for data_dict in data_dicts:

        for key, value in data_dict.items():

            if key in FEATURE_NAMES or key == 'asking_price_euros':

                if value < min_dict.get(key, float_info.max):
                    min_dict[key] = value
    
                if value > max_dict.get(key, float_info.min):
                    max_dict[key] = value
    
    print('Found max dictionary:\n{}'.format(max_dict))
    print('Found min dictionary:\n{}'.format(min_dict))

    # Now map linearly to [-1, 1]
    normed_data_dicts = list()
    for data_dict in data_dicts:

        normed_dict = dict()
        normed_dict['file_path'] = data_dict['file_path']

        for key, value in data_dict.items():

            if key in FEATURE_NAMES or key == 'asking_price_euros':

                normed_dict[key] = map_to_min1_plus1(
                    value,
                    min_dict[key],
                    max_dict[key])

            elif key == 'builder_name':
                normed_dict[key] = data_dict[key]

        normed_data_dicts.append(normed_dict)

    return normed_data_dicts, min_dict, max_dict


def write_data_to_file(save=True):

    """
    This function writes all needed data to file, ready for analysis.
    For the format, see above.

    Information written to file:
    FEATURE_NAMES
    BUILDER_NAMES
    input_data
    target_data

    If save is True, then the result will be written to disk,
    otherwise not.
    """

    data_dicts = get_boat_data_cr_dicts_from_files()

    print('\nCheck 1:')
    print(data_dicts[1:5])

    data_dicts = remove_outliers(data_dicts)

    print('\nCheck 2:')
    print(data_dicts[1:5])

    data_dicts, averages = substitute_averages(data_dicts)

    data_dicts = transform_data(data_dicts)

    print('\nCheck 3:')
    print(data_dicts[1:5])

    data_dicts, min_dict, max_dict = normalise_to_min1_plus1(data_dicts)

    print('\nCheck 4:')
    print(data_dicts[1:5])

    input_data, target_data, file_paths = get_boat_data_cr(data_dicts)

    print('\nCheck 5:')
    print('{}: {}'.format(file_paths[0], input_data[:, [0]]))
    print('{}: {}'.format(file_paths[1], input_data[:, [1]]))
    M = int(len(file_paths) / 2)
    print('{}: {}'.format(file_paths[M], input_data[:, [M]]))

    print('input_data shape: \n{}'.format(input_data.shape))
    print('target_data.shape: \n{}'.format(target_data.shape))
    print('files: {}\n'.format(len(file_paths)))

    if save:
        print('Saving data to: {}'.format(DATA_DIR_OUT))
        np.save('{}/{}'.format(DATA_DIR_OUT, 'feature_names'), FEATURE_NAMES)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'builder_names'), BUILDER_NAMES)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'averages'), averages)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'input_data'), input_data)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'target_data'), target_data)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'file_paths'), file_paths)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'min_dict'), min_dict)
        np.save('{}/{}'.format(DATA_DIR_OUT, 'max_dict'), max_dict)
    else:
        print('Not saving data')

write_data_to_file(True)
