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


import os  
import numpy as np

DATA_DIR_IN = '/home/ytsboe/data/boats/raw_data'
DATA_DIR_OUT = '/home/ytsboe/data/boats/computer_readable'
MIN_BUILDERS_COUNT = 30
FEATURE_NAMES = ('length_over_all_meters',
                 'width_meters',
                 'build_year',
                 'draft_meters',
                 'displaces_kgs',
                 'ballast_kgs',
                 'engine_build_year')


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
    for builder, count in builders_dict_raw.iteritems():

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
    for builder, count in builders_dict.iteritems():

        total_boats_count += count

        print('{}: {}'.format(count, builder))

    print('total_boats_count = {}'.format(total_boats_count))

    return builders_dict


BUILDERS_COUNT_DICT = get_builders_count_dict()
print('BUILDERS_COUNT_DICT = {}'.format(BUILDERS_COUNT_DICT))

BUILDER_NAMES = BUILDERS_COUNT_DICT.keys()


def get_val(line):

    val = line.split(':')[1]
    if 'non' in val.lower():
        val = None
    else:
        val = float(val)

    return val


def get_boat_data_dicts():

    """
    This function fetches the boat data in a dictionary for each boat.
    It returns a list of boat data dicts.
    """

    dict_list = list()

    for file in get_file_full_paths():

        with open(file) as f:

            # print('Starting file: {}'.format(file))

            boat_data = dict()

            for line in f:

                for key in FEATURE_NAMES:

                    if key in line:

                        # Don't confuse engine_build_year with build_year
                        if key == 'build_year' and 'engine' in line:
                            continue
                        
                        boat_data[key] = get_val(line)

                if 'asking_price_euros' in line:
                    boat_data['asking_price_euros'] = get_val(line)

                if 'builder_name' in line:
                    boat_data['builder_name'] = line.split(':')[1].split()[0]

            # print('Got boat data: {}'.format(boat_data))
            dict_list.append(boat_data)

    return dict_list

    
def get_boat_data(data_dicts):

    """
    This function creates the rows of the data matrix
    as defined above.

    It substitutes averages for numerical values, which are missing.

    It extens the numerical data with the sparse array for the builders.

    It returns a matrix, of which the colums are the input vectors for training.
    """

    # Collect sums and counts, preparing to calculate averages
    sum_dict = dict()
    count_dict = dict()

    for data_dict in data_dicts:

        for key, value in data_dict.iteritems():

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

    # Substitute None's with averages
    completed_dicts = list()
    for data_dict in data_dicts:
        subst_dict = dict()
        for key, value in data_dict.iteritems():
            if key in FEATURE_NAMES or key == 'asking_price_euros':
                if value:
                    subst_dict[key] = value
                else:
                    subst_dict[key] = av_dict[key]

            elif key == 'builder_name':

                keys_list = BUILDERS_COUNT_DICT.keys()
                sparse_list = np.zeros(len(keys_list))
                if value in BUILDERS_COUNT_DICT.keys():
                    sparse_list[keys_list.index(value)] = 1.
                else:
                    sparse_list[keys_list.index('None')] = 1.
                subst_dict[key] = sparse_list

        completed_dicts.append(subst_dict)

    # print('Found {} completed dicts:'.format(len(completed_dicts)))
    # for c_dict in completed_dicts:
    #     for k,v in c_dict.iteritems():
    #         print('  {}: {}'.format(k, v))
    #     print('----')

    # Finally, we can now create the output matrices
    boat_count = len(completed_dicts)
    col_count = len(FEATURE_NAMES) + len(BUILDERS_COUNT_DICT.keys())

    print('boat_count = {}'.format(boat_count))
    print('col_count  = {}'.format(col_count))

    input_data = np.zeros(boat_count * col_count)
    target_data = np.zeros(boat_count)

    for i in range(len(completed_dicts)):

        col0 = i * col_count

        data_dict = completed_dicts[i]
        print('data_dict:')
        for key in FEATURE_NAMES:
            print('  {}: {}'.format(key, data_dict[key]))
        print('  builder_name: {}'.format(data_dict['builder_name']))

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

    return np.transpose(input_data), np.transpose(target_data), 


def write_data_to_file():

    """
    This function writes all needed data to file, ready for analysis.
    For the format, see above.

    Information written to file:
    FEATURE_NAMES
    BUILDER_NAMES
    input_data
    target_data
    """

    data_dicts = get_boat_data_dicts()

    input_data, target_data = get_boat_data(data_dicts)

    # print('input_data: \n{}'.format(input_data))
    # print('target_data: \n{}'.format(target_data))

    np.save('{}/{}'.format(DATA_DIR_OUT,'feature_names'), FEATURE_NAMES)
    np.save('{}/{}'.format(DATA_DIR_OUT,'builder_names'), BUILDER_NAMES)
    np.save('{}/{}'.format(DATA_DIR_OUT,'input_data'), input_data)
    np.save('{}/{}'.format(DATA_DIR_OUT,'target_data'), target_data)

write_data_to_file()

    
    
