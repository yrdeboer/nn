import numpy as np

# In this file we read the computer readable data and reproduce
# the averages and builder counts to compare with the earlier results.

DATA_DIR_CR = '/home/ytsboe/data/boats/computer_readable'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))

print('\nAverages:')
for i in range(len(feature_names)):

    av = np.mean(input_data[i])
    print('{}: {}'.format(feature_names[i], av))

print('\nSums:')
row1 = len(feature_names)
row2 = row1 + len(builder_names)
sums = np.sum(input_data[row1:row2], axis=1)
for i in range(len(builder_names)):
    print('  {}: {}'.format(builder_names[i], sums[i]))

