import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline

def plot_units():
    subj = '017'
    chan = 'RA8'
    cluster = 1
    peaks = np.load(
        fr"C:\repos\NirsLabProject\NirsLabProject\data\products\p{subj}_MTL\bipolar_model\spikes\peaks-{chan[:-1]}1.npz.npy").flatten()
    spikes = pd.read_csv(fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}.csv", header=None)[0].to_list()

    unit = np.zeros(spikes[-1])
    unit[spikes[:-1]] = 1
    all_trials = np.zeros(shape=(len(peaks), 1000))
    trials_hist = np.zeros(shape=(len(peaks), 1000))
    for i, peak in enumerate(peaks):
        curr = np.array(unit[peak - 500: peak + 500])
        all_trials[i] = curr
        curr_hist = np.arange(1000)
        curr_hist[curr == 0] = 0
        trials_hist[i] = curr_hist

    fig, ax = plt.subplots(2, 1, figsize=[10,5])

    # Loop to plot raster for each trial
    y_index = 0
    for trial in range(len(all_trials)):
        spike_times = [i for i, x in enumerate(all_trials[trial]) if x == 1]
        # if len(spike_times) > 0:
        ax[0].vlines(spike_times, trial - 4, trial + 4)
            # y_index += 1

    ax[0].set_xlim([0, 1000])
    ax[0].set_ylim([0, len(all_trials)])
    ax[1].set_xlim([0, 1000])
    ax[1].set_xlabel('Time (ms)')


    # specify tick marks and label y axis
    # ax.set_ylabel('Trial Number')

    ax[0].set_title(f'{subj}-{chan}-unit{cluster}')
    ax[1].set_title('Peri-Stimulus Time Histogram (PSTH)')
    ax[1].set_ylabel('Firing Rate (Hz)')

    # Draw the PSTH
    bin_size = 50
    counts, bins = np.histogram(trials_hist[trials_hist != 0], bins=bin_size)
    ax[1].hist(bins[:-1], bins, weights=counts/(len(all_trials) * bin_size), alpha=0.5)

    # smoothing
    x = np.arange(0, 1000, int(1000/bin_size))
    y = counts / (len(all_trials) * bin_size)
    X_Y_Spline = make_interp_spline(x, y)

    # Returns evenly spaced numbers over a specified interval.
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    ax[1].plot(X_, Y_, color='black')

    # sns.histplot(trials_hist[trials_hist != 0], ax=ax[1], bins=bin_size, kde=True)

    # mark spike
    ax[0].axvline(500, color='r')
    ax[1].axvline(500, color='r')
    plt.tight_layout()
    plt.savefig(fr"D:\Hanna\D{subj}\units\{chan}_unit{cluster}.png")
    plt.show()
    print()

plot_units()