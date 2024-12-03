from curses.ascii import ctrl
from distutils.log import error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
from scipy.stats import ttest_ind
import scipy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants

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

ERR = 'ci' # std, ci, sem
INCLUDE = 'all' # 'linear', 'circular', 'all'
print(INCLUDE)

ctrl_errors = []
ctrl2_errors = []
ctrl3_errors = []
tx_errors = []

for BOT_ID in constants.BOT_IDs:

    os.makedirs('plots/'+BOT_ID+'/heading/',exist_ok=True)

    path_to_heading_data = 'heading_errors/'+BOT_ID+'.csv'

    df = pd.read_csv(path_to_heading_data)

    if INCLUDE=='linear':
        if BOT_ID in constants.LINEARS: # add only linears
            ctrl_errors.append(np.nanmean(df[df['tx']=='ctrl']['normalized_error']))
            ctrl2_errors.append(np.nanmean(df[df['tx']=='ctrl2']['normalized_error']))
            ctrl3_errors.append(np.nanmean(df[df['tx']=='ctrl3']['normalized_error']))
            tx_errors.append(np.nanmean(df[df['tx']=='tx']['normalized_error']))
    elif INCLUDE=='circular': 
        if BOT_ID in constants.CIRCULARS: # add only circulars 
            ctrl_errors.append(np.nanmean(df[df['tx']=='ctrl']['normalized_error']))
            ctrl2_errors.append(np.nanmean(df[df['tx']=='ctrl2']['normalized_error']))
            ctrl3_errors.append(np.nanmean(df[df['tx']=='ctrl3']['normalized_error']))
            tx_errors.append(np.nanmean(df[df['tx']=='tx']['normalized_error']))
    
    else: # both - include all of the data
        # Compute mean errors for each bot
        ctrl_errors.append(np.nanmean(df[df['tx']=='ctrl']['normalized_error']))
        ctrl2_errors.append(np.nanmean(df[df['tx']=='ctrl2']['normalized_error']))
        ctrl3_errors.append(np.nanmean(df[df['tx']=='ctrl3']['normalized_error']))
        tx_errors.append(np.nanmean(df[df['tx']=='tx']['normalized_error']))

# HISTOGRAMS
# plt.hist(ctrl_errors)
# plt.hist(ctrl2_errors)
# plt.hist(ctrl3_errors)
# plt.hist(tx_errors)
# plt.legend(constants.TX_LABELS)
# plt.xlabel('Error')
# plt.ylabel('# Runs')

# SCATTER PLOTS
xs = np.arange(1,5)
bars = [np.mean(ctrl_errors), np.mean(ctrl2_errors), np.mean(ctrl3_errors), np.mean(tx_errors)]
stds = [np.std(ctrl_errors), np.std(ctrl2_errors), np.std(ctrl3_errors), np.std(tx_errors)]
sem = [scipy.stats.sem(ctrl_errors), scipy.stats.sem(ctrl2_errors), scipy.stats.sem(ctrl3_errors), scipy.stats.sem(tx_errors)]

CI_95_ctrl = 1.96 * (np.std(ctrl_errors)/np.sqrt(len(ctrl_errors)))
CI_95_ctrl2 = 1.96 * (np.std(ctrl2_errors)/np.sqrt(len(ctrl2_errors)))
CI_95_ctrl3 = 1.96 * (np.std(ctrl3_errors)/np.sqrt(len(ctrl3_errors)))
CI_95_tx = 1.96 * (np.std(tx_errors)/np.sqrt(len(tx_errors)))
CI_95 = [CI_95_ctrl, CI_95_ctrl2, CI_95_ctrl3, CI_95_tx]

if ERR=='std':
    err=stds
elif ERR == 'ci':
    err=CI_95

fig, ax = plt.subplots()
plt.bar(xs,bars, color=(0.55,0.55,0.55,0.75))
plt.errorbar(xs, bars, yerr=err, fmt='o', capsize=5, color = (0.25,0.25,0.25))
ax.set_xticks(xs)
ax.set_xticklabels(constants.TX_LABELS, fontsize=13)
plt.ylabel('Error in initial heading', fontsize=17)
plt.yticks(fontsize=15)

# plt.title(INCLUDE)

# Compared to control
stat, ctrl_ctrl2_p = ttest_ind(ctrl_errors, ctrl2_errors, nan_policy='omit')
stat, ctrl_ctrl3_p = ttest_ind(ctrl_errors, ctrl3_errors, nan_policy='omit')
stat, ctrl_tx_p = ttest_ind(ctrl_errors, tx_errors, nan_policy='omit')

print(ctrl_ctrl2_p, ctrl_ctrl3_p, ctrl_tx_p)

maxy = [sum(x) for x in zip(bars, err)]

# Annotate plot
h=.01
# start = 0.8 #if ERR=='std' else 0.6
xs = np.arange(1,5)
# ctrl vs. ctrl2 (+shape/-CD) annotation 
# y1 = max(np.nanmax(ctrl_errors), np.nanmax(ctrl2_errors))
y1= np.max((maxy[0]+h, maxy[1])) + 0.01
plt.plot([xs[0], xs[0], xs[1], xs[1]], [y1, y1+h, y1+h, y1], lw=1.5, c='k')
txt1 = significance_level(ctrl_ctrl2_p)
plt.text((xs[0]+xs[1])*.5, y1+h, txt1, ha='center', va='bottom', color='k',fontsize=15)

# # ctrl vs. ctrl3 (-shape/+CD) annotation 
# y2 = max(y1, np.nanmax(ctrl3_errors))
y2=np.max((maxy[0]+h, maxy[1]+h, maxy[2])) + 0.075
plt.plot([xs[0], xs[0], xs[2], xs[2]], [y2, y2+h, y2+h, y2], lw=1.5, c='k')
txt2 = significance_level(ctrl_ctrl3_p)
plt.text((xs[0]+xs[2])*.5, y2+h, txt2, ha='center', va='bottom', color='k', fontsize=15)

# # ctrl vs. tx (+shape/+CD) annotation 
# y3 = max(y2, np.nanmax(tx_errors))
y3=np.max((maxy[0]+h, maxy[1]+h, maxy[2]+h, maxy[3])) + 0.15
plt.plot([xs[0], xs[0], xs[3], xs[3]], [y3, y3+h, y3+h, y3], lw=1.5, c='k')
txt3 = significance_level(ctrl_tx_p)
plt.text((xs[0]+xs[3])*.5, y3+h, txt3, ha='center', va='bottom', color='k', fontsize=15)

print(ctrl_ctrl2_p, ctrl_ctrl3_p, ctrl_tx_p)

# plt.title('Mean Error with 95% C.I. - {}'.format(INCLUDE))

plt.savefig('../../../Desktop/summary.png', bbox_inches='tight', dpi=300)
plt.savefig('plots/summary/heading_error_{}_errbars_{}.png'.format(INCLUDE, ERR), dpi=300, bbox_inches='tight')
# plt.show()
