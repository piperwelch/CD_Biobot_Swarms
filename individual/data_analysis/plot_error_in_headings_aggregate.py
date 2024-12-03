import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

AGGREGATE = False # Linear and circular data together

def significance_level(p):
    # print(p)
    if p<0.001:
        return "***"
    elif p<0.01:
        return "**"
    elif p<0.05:
        return "*"
    else:
        return "n.s."

def scatter_plot(data, title=''):

    # Order that the data comes in should be: ctrl, ctrl2, ctrl3, tx

    labels = ['-shape/-CD','+shape/-CD', '-shape/+CD','+shape/+CD']
    xs = np.arange(1,5)
    bars = [np.nanmean(data[0]), np.nanmean(data[1]), np.nanmean(data[2]), np.nanmean(data[3])]
    stds = [np.nanstd(data[0]), np.nanstd(data[1]), np.nanstd(data[2]), np.nanstd(data[3])]

    print(bars)
    print(stds)

    fig, ax = plt.subplots()
    plt.scatter(xs,bars)
    plt.errorbar(xs, bars, yerr=stds, fmt='o', capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)

    ylab = plt.ylabel(title)
    # ylab.set_position((1, 0.5))

    # Significance 

    # Compared to control
    stat, ctrl_ctrl2_p = ttest_ind(data[0], data[1], equal_var=False, nan_policy='omit')
    stat, ctrl_ctrl3_p = ttest_ind(data[0], data[2], equal_var=False, nan_policy='omit')
    stat, ctrl_tx_p = ttest_ind(data[0], data[3], equal_var=False, nan_policy='omit')

    print(ctrl_ctrl2_p, ctrl_ctrl3_p, ctrl_tx_p)

    # # Comparing treatments
    # stat, ctrl2_ctrl3_p = ttest_ind(ctrl2_chirality, ctrl3_chirality, equal_var=False)
    # stat, ctrl2_tx_p = ttest_ind(ctrl2_chirality, tx_chirality, equal_var=False)
    # stat, ctrl3_tx_p = ttest_ind(ctrl3_chirality, tx_chirality, equal_var=False)

    plt.show()

in_silico_headings_files = glob('heading_errors/*.csv')

# Linears
ctrl_error_str = []
ctrl2_error_str = []
ctrl3_error_str = []
tx_error_str = []

ctrl_error_std_str = []
ctrl2_error_std_str = []
ctrl3_error_std_str = []
tx_error_std_str = []

# Circulars

ctrl_error_circ = []
ctrl2_error_circ = []
ctrl3_error_circ = []
tx_error_circ = []

ctrl_error_std_circ = []
ctrl2_error_std_circ = []
ctrl3_error_std_circ = []
tx_error_std_circ = []

for filename in in_silico_headings_files:
    BOT_ID = filename.split('/')[-1].split('.')[0]

    df = pd.read_csv(filename)

    for row in df.iterrows():
        TX = row[1]['treatment'].split('_')[-1]

        error = float(row[1]['error_angle'])
        std = float(row[1]['std'])

        if 'bot' in BOT_ID: # Linear
            if TX == "ctrl":
                ctrl_error_str.append(error)
                ctrl_error_std_str.append(std)
            elif TX == "ctrl2":
                ctrl2_error_str.append(error)
                ctrl2_error_std_str.append(std)
            elif TX == "ctrl3":
                ctrl3_error_str.append(error)
                ctrl3_error_std_str.append(std)
            elif TX == "tx":
                tx_error_str.append(error)
                tx_error_std_str.append(std)
        else: # Circular
            if TX == "ctrl":
                ctrl_error_circ.append(error)
                ctrl_error_std_circ.append(std)
            elif TX == "ctrl2":
                ctrl2_error_circ.append(error)
                ctrl2_error_std_circ.append(std)
            elif TX == "ctrl3":
                ctrl3_error_circ.append(error)
                ctrl3_error_std_circ.append(std)
            elif TX == "tx":
                tx_error_circ.append(error)
                tx_error_std_circ.append(std)

if AGGREGATE:
    pass
    # ctrl_mean_err = np.nanmean(ctrl_error)
    # ctrl_std_err = np.nanstd(ctrl_error)
    # ctrl_mean_std = np.nanmean(ctrl_error_std)

    # ctrl2_mean_err = np.nanmean(ctrl2_error)
    # ctrl2_std_err = np.nanstd(ctrl2_error)
    # ctrl2_mean_std = np.nanmean(ctrl2_error_std)

    # ctrl3_mean_err = np.nanmean(ctrl3_error)
    # ctrl3_std_err = np.nanmean(ctrl3_error)
    # ctrl3_mean_std = np.nanmean(ctrl3_error_std)

    # tx_mean_err = np.nanmean(tx_error)
    # tx_std_err = np.nanmean(tx_error)
    # tx_mean_std = np.nanmean(tx_error_std)

else:

    # Linear - compute means and plot
    data_mean_err_str = [ctrl_error_str, ctrl2_error_str, ctrl3_error_str, tx_error_str]
    data_std_err_str = [ctrl_error_str, ctrl2_error_str, ctrl3_error_str, tx_error_str]
    data_mean_std_str = [ctrl_error_std_str, ctrl2_error_std_str, ctrl3_error_std_str, tx_error_std_str]

    scatter_plot(data_mean_err_str)

    # Circular - compute means and plot
    data_mean_err_circ = [ctrl_error_circ, ctrl2_error_circ, ctrl3_error_circ, tx_error_circ]
    data_std_err_circ = [ctrl_error_circ, ctrl2_error_circ, ctrl3_error_circ, tx_error_circ]
    data_mean_std_circ = [ctrl_error_std_circ, ctrl2_error_std_circ, ctrl3_error_std_circ, tx_error_std_circ]

    scatter_plot(data_mean_err_circ)