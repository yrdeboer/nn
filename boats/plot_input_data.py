import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# In this file we create the plots we can use to clean the data, if necessary
#
# To start off with, we plot all feature distributions

DATA_DIR_CR = '/home/ytsboe/data/boats/computer_readable'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))

HIST_BIN_COUNT = 25

def plot_feature_distributions():

    pp = PdfPages('feature_distributions.pdf')

    plt.hist(np.transpose(target_data), HIST_BIN_COUNT)
    plt.title('asking_price_euros')
    plt.savefig(pp, format='pdf')
    plt.close()

    row_count = len(feature_names)
    for i in range(row_count):

        plt.hist(input_data[i], HIST_BIN_COUNT)
        plt.title(feature_names[i])
        plt.savefig(pp, format='pdf')
        plt.close()

    pp.close()


def plot_feature_asking_price_scatter_plots():

    """
    This function plots all features + asking price
    scatter plots.
    """

    pp = PdfPages('feature_asking_price_scatter_plots.pdf')

    row_count = len(feature_names)
    for i in range(row_count):

        plt.scatter(input_data[i], np.transpose(target_data))
        plt.title(feature_names[i])
        plt.savefig(pp, format='pdf')
        plt.close()

    pp.close()


# plot_feature_distributions()
plot_feature_asking_price_scatter_plots()
