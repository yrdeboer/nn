import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# In this file we create the plots we can use to clean the data, if necessary
#
# To start off with, we plot all feature distributions

DATA_DIR_CR = 'computer_readable_data'

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


def plot_feature_scatter_plots():

    """
    This function plots all feature scatter plots
    with all other features.
    """

    pp = PdfPages('feature_scatter_plots.pdf')

    row_count = len(feature_names)

    for i in range(row_count):

        for j in range(i+1, row_count):

            # plt.subplot(row_count, row_count, i*row_count + j)
            plt.scatter(input_data[i], input_data[j])
            title = '{} (vert) vs. {} (horz)'.format(feature_names[j], feature_names[i])
            print('Plotting {}'.format(title))
            plt.title(title)
            plt.savefig(pp, format='pdf')    
            plt.close()


    pp.close()


def plot_asking_price_for_builder_name():

    """
    This function plots in one plot the asking price distribution for each
    builder.
    """

    pp = PdfPages('asking_price_builder_names.pdf')

    prices_all = target_data.reshape(target_data.shape[1])
    price_count = len(prices_all)

    prices_avg = np.zeros(len(builder_names))

    len1 = len(feature_names)
    len2 = len1 + len(builder_names)
    for i in range(len1, len2):

        bld_count = np.sum(input_data[i])
        prices_bld = np.zeros(bld_count)
        k = 0
        for j in range(price_count):
            if input_data[i][j] > 0.5:
                prices_bld[k] = prices_all[j]
                k += 1

        prices_avg[i-len1] = np.mean(prices_bld)

        bin_count = bld_count / 10
        if bin_count == 0:
            bin_count = 2

        print('prices_bld = {} shape={}'.format(prices_bld, prices_bld.shape))

        plt.hist(prices_bld, bin_count)
        title = 'Asking prices ({0:}, avg={1:.2f}) hist for builder: {2:}'.format(
            bld_count,
            prices_avg[i-len1],
            builder_names[i-len1])
        print(title)
        plt.title(title)
        plt.savefig(pp, format='pdf')    
        plt.close()

    pp.close()

    print('Average average price = {}'.format(np.mean(prices_avg)))


plot_feature_distributions()
plot_feature_asking_price_scatter_plots()
plot_feature_scatter_plots()
# plot_asking_price_for_builder_name()
