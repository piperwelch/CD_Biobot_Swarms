import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants

# BOT_ID = 'Run4group0subject4'

for BOT_ID in constants.BOT_IDs:

    os.makedirs('plots/'+BOT_ID+'/heading/',exist_ok=True)

    path_to_heading_data = 'heading_errors/'+BOT_ID+'.csv'

    df = pd.read_csv(path_to_heading_data)

    ctrl_heading_errors = np.asarray(df[df['tx']=='ctrl']['normalized_error'])
    ctrl2_heading_errors = np.asarray(df[df['tx']=='ctrl2']['normalized_error'])
    ctrl3_heading_errors = np.asarray(df[df['tx']=='ctrl3']['normalized_error'])
    tx_heading_errors = np.asarray(df[df['tx']=='tx']['normalized_error'])

    # HISTOGRAMS
    # plt.hist(ctrl_heading_errors)
    # plt.hist(ctrl2_heading_errors)
    # plt.hist(ctrl3_heading_errors)
    # plt.hist(tx_heading_errors)
    # plt.legend(constants.TX_LABELS)
    # plt.xlabel('Error')
    # plt.ylabel('# Runs')
    # plt.savefig('plots/'+BOT_ID+'/heading/heading_errors_hist.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # SCATTER PLOTS
    xs = np.arange(1,5)
    bars = [np.nanmean(ctrl_heading_errors), np.nanmean(ctrl2_heading_errors), np.nanmean(ctrl3_heading_errors), np.nanmean(tx_heading_errors)]
    stds = [np.nanstd(ctrl_heading_errors), np.nanstd(ctrl2_heading_errors), np.nanstd(ctrl3_heading_errors), np.nanstd(tx_heading_errors)]

    fig, ax = plt.subplots()
    plt.scatter(xs,bars)
    plt.errorbar(xs, bars, yerr=stds, fmt='o', capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(constants.TX_LABELS)
    plt.ylabel('Error')
    plt.savefig('plots/'+BOT_ID+'/heading/heading_errors_mean_scatter.png', dpi=300, bbox_inches='tight')

    plt.title(BOT_ID)
    # plt.show()
    plt.close()