import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import glob


def plot_units(subj='018', chan='RMH7', cluster=1, peaks_file=None):
    if peaks_file is None:
        peaks_file = chan[:-1]
    peaks = np.load(
        fr"C:\repos\NirsLabProject\NirsLabProject\data\products\p{subj}_MTL\bipolar_model\spikes\peaks-{peaks_file}1.npz.npy").flatten()
    spikes = pd.read_csv(fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}.csv", header=None)[0].to_list()

    unit = np.zeros(spikes[-1])
    unit[spikes[:-1]] = 1
    all_trials = np.zeros(shape=(len(peaks), 1000))
    trials_hist = np.zeros(shape=(len(peaks), 1000))
    for i, peak in enumerate(peaks):
        curr = np.array(unit[peak - 500: peak + 500])
        if len(curr) > 0:
            all_trials[i] = curr
            curr_hist = np.arange(1000)
            curr_hist[curr == 0] = 0
            trials_hist[i] = curr_hist
        else:
            all_trials = np.delete(all_trials, np.s_[i:], axis=0)
            trials_hist = np.delete(trials_hist, np.s_[i:], axis=0)
            break


    fig, ax = plt.subplots(2, 1, figsize=[10,5])

    # Loop to plot raster for each trial
    line_size = 0.5 if len(all_trials) < 100 else 1
    line_size = 4 if len(all_trials) > 400 else 1
    for trial in range(len(all_trials)):
        spike_times = [i for i, x in enumerate(all_trials[trial]) if x == 1]
        ax[0].vlines(spike_times, trial - line_size, trial + line_size)

    ax[0].set_xlim([0, 1000])
    ax[0].set_ylim([0, len(all_trials)])
    ax[1].set_xlim([0, 1000])
    ax[1].set_xlabel('Time (ms)')


    # specify tick marks and label y axis
    # ax.set_ylabel('Trial Number')

    ax[0].set_title(f'{subj}-{chan}-unit{cluster}-{peaks_file} peaks')
    ax[1].set_title('Peri-Stimulus Time Histogram (PSTH)')
    ax[1].set_ylabel('Firing Rate (Hz)')

    # Draw the PSTH
    bin_size = 50
    counts, bins = np.histogram(trials_hist[trials_hist != 0], bins=bin_size)
    ax[1].hist(bins[:-1], bins, weights=(counts/(len(all_trials) * bin_size)) * 1000, alpha=0.5)

    # smoothing
    x = np.arange(0, 1000, int(1000/bin_size))
    y = (counts / (len(all_trials) * bin_size)) * 1000
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    ax[1].plot(X_, Y_, color='black')

    # sns.histplot(trials_hist[trials_hist != 0], ax=ax[1], bins=bin_size, kde=True)

    # mark spike
    ax[0].axvline(500, color='r')
    ax[1].axvline(500, color='r')
    plt.tight_layout()
    file_name = fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}.png" if peaks_file is None else fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}_{peaks_file}peaks.png"
    plt.savefig(file_name)
    plt.show()
    print()


def plot_combine_units(subj, chan_lst):
    # draw each unit in different color
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    peaks = np.load(
        fr"C:\repos\NirsLabProject\NirsLabProject\data\products\p{subj}_MTL\bipolar_model\spikes\peaks-{chan_lst[0][0][:-1]}1.npz.npy").flatten()
    # all_units = np.zeros(shape=(len(chan_lst) * len(peaks), 1000))
    fig, ax = plt.subplots(2, 1, figsize=[16, 10])

    for color, (chan, cluster) in enumerate(chan_lst):
        spikes = pd.read_csv(fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}.csv", header=None)[0].to_list()

        unit = np.zeros(spikes[-1])
        unit[spikes[:-1]] = 1
        all_trials = np.zeros(shape=(len(peaks), 1000))
        trials_hist = np.zeros(shape=(len(peaks), 1000))
        for i, peak in enumerate(peaks):
            curr = np.array(unit[peak - 500: peak + 500])
            if len(curr) > 0:
                all_trials[i] = curr
                curr_hist = np.arange(1000)
                curr_hist[curr == 0] = 0
                trials_hist[i] = curr_hist
            else:
                all_trials = np.delete(all_trials, np.s_[i:], axis=0)
                trials_hist = np.delete(trials_hist, np.s_[i:], axis=0)
                break

        # Loop to plot raster for each trial
        line_size = 4 if len(all_trials) > 400 else 1
        for trial in range(len(all_trials)):
            spike_times = [i for i, x in enumerate(all_trials[trial]) if x == 1]
            ax[0].vlines(spike_times, trial - line_size, trial + line_size, color=colors[color])

        bin_size = 50
        counts, bins = np.histogram(trials_hist[trials_hist != 0], bins=bin_size)
        # ax[1].hist(bins[:-1], bins, weights=counts / (len(all_trials) * bin_size), alpha=0.5)

        # smoothing
        x = np.arange(0, 1000, int(1000 / bin_size))
        y = (counts / (len(all_trials) * bin_size)) * 1000
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        ax[1].plot(X_, Y_, color=colors[color])

    ax[0].set_xlim([0, 1000])
    ax[0].set_ylim([0, len(all_trials)])
    ax[1].set_xlim([0, 1000])
    ax[1].set_xlabel('Time (ms)')
    ax[0].axvline(500, color='r')
    ax[1].axvline(500, color='r')
    ax[0].set_title(f'{subj}-{chan[:-1]}')
    ax[1].set_title('Peri-Stimulus Time Histogram (PSTH)')
    ax[1].set_ylabel('Firing Rate (Hz)')

    plt.tight_layout()
    plt.savefig(fr"D:\Hanna\D{subj}\units\{subj}-{chan[:-1]}.png")

subj = {'017':[('RA3', 1), ('RA4', 1), ('RA8', 1), ('RA8', 2), ('LPHG1', 1), ('LPHG8', 1), ('LPHG8', 2)],
        '018':[('RA5', 1), ('RA8', 1), ('RMH2', 1), ('RMH4', 1), ('RMH6', 1), ('RMH6', 3), ('RMH7', 1)],
        '025':[('LA3', 1), ('LA3', 2), ('LA3', 3)]}

# plot_units()
subj = '018'
subj_files_list = glob.glob(f'D:\\Hanna\\D{subj}\\units\\*.csv')
for file in subj_files_list:
    chan = file.split('\\')[-1].split('_')[0]
    cluster = file.split('\\')[-1].split('_')[1].split('.')[0][-1]
    plot_units(subj, chan, cluster, 'RMH')

# plot_units('017', 'LOF7', 1, 'RA')
# for s in subj:
# s = '018'
# for c in subj[s]:
#     plot_units(s, c[0], c[1])

# plot_combine_units('018', [x for x in subj['018'] if x[0].startswith('RMH')])