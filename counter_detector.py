import mne
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
from scipy import stats, signal
import itertools
import pandas as pd
from datetime import datetime
# mpl.use('Agg')

# channels names: AH= anterior hippocampus, EC=entorhinal cortex, PHG = parahippocampal gyrus
channels_names = ['LAH', 'RAH', 'LEC', 'REC', 'LPHG', 'RPHG']

# general constants
min_distance = 200  # minimal distance for 'different' spikes - in miliseconds


# constants for detection based on frequency analysis - the thresholds for the standard deviation are based on the
# papers Andrillon et al(envelope condition) and Starestina et al(other conditions)
threshold_env = 8  # threshold in standard deviations for the envelope after bandpass (HP)
threshold_amp = 5  # threshold in standard deviations for the amplitude
threshold_grad = 5  # threshold in standard deviations for the gradient
thresholdConjAmp = 3  # threshold in standard deviations for the amplitude for the conjunction of amp & grad condition
thresh_amp_grad = 3  # threshold in standard deviations for the gradient for the conjunction of amp & grad condition
thresh_amp_env = 3  # threshold in standard deviations for the HP for the conjunction of amp & HP condition
use_env = True
use_amp = True
use_grad = True
use_amp_grad = True
use_amp_env = True
is_disjunction = True
block_size_sec = 10  # filter and find peaks at blocks of X seconds - based on Andrillon et al

# the bandpass range is based on Andrillon et al
low_pass = 50
high_pass = 150

min_spike_length_ms = 5  # a spike is detected if there are points for X ms passing the threshold - in ms, based on Andrillon et al
conditions_true_if_any = False
percentageOfNansAllowedAroundSpike = 0.1  # how many NaNs are allowed in the vicinity of the spike(vicinity=minDistSpikes / 2 before and after)
HighBandPassScore = 11  # this is for debugging - finding especially high STDs for HP
common_spikes_index = []


# format: [spike_id, threshold_type, first_index, last_index, max_index, max_amp, duration(calc once for all after convert to df), sleep_stage?]
spikes_list = []
id_count = itertools.count(1)
spikes_counter = []


def ranges(nums):
    from itertools import groupby
    from operator import itemgetter
    return_list = []
    for k, g in groupby(enumerate(nums), lambda i_x: i_x[0] - i_x[1]):
        return_list.append(list(map(itemgetter(1), g)))

    return return_list


def detect(raw_channel, sampling_rate, plot=True):
    data = raw_channel.get_data()[0]
    channel_counter = [channel, 0, 0, 0]

    points_in_block = block_size_sec * sampling_rate
    number_of_blocks = math.floor(len(data) / points_in_block)

    for i in range(number_of_blocks):
        curr_block = data[i * points_in_block: (i + 1) * points_in_block]

        # check amplitude threshold
        if use_amp or use_amp_env or use_amp_grad:
            z_score_amp = stats.zscore(curr_block)
            points_above_thresh_amp = z_score_amp[z_score_amp > threshold_amp]
            # get indexes from z_score values and add offset of the current block
            if len(points_above_thresh_amp) > 0:
                index_above_threshold_amp = (z_score_amp > threshold_amp).nonzero()[0] + i * points_in_block
                channel_counter[1] += len(ranges(index_above_threshold_amp))

        # check gradient threshold
        if use_grad or use_amp_grad:
            gradient_diff = np.diff(curr_block)
            z_score_grad = stats.zscore(np.insert(gradient_diff, 0, 0))
            points_above_thresh_grad = z_score_grad[z_score_grad > threshold_grad]
            if len(points_above_thresh_grad) > 0:
                index_above_threshold_grad = (z_score_grad > threshold_grad).nonzero()[0] + i * points_in_block
                channel_counter[2] += len(ranges(index_above_threshold_grad))

        # check envelope threshold
        if use_env or use_amp_env:
            filtered_block = mne.filter.filter_data(curr_block, sampling_rate, low_pass, high_pass)
            env_block = abs(signal.hilbert(filtered_block))
            z_score_env = stats.zscore(env_block)
            points_above_thresh_env = z_score_env[z_score_env > threshold_env]
            if len(points_above_thresh_env) > 0:
                index_above_threshold_env = (z_score_env > threshold_env).nonzero()[0] + i * points_in_block
                channel_counter[3] += len(ranges(index_above_threshold_env))

    spikes_counter.append(channel_counter)

    return True


start_time = datetime.now()
mne.set_log_level(False)
save_csv = True
edf = 'C:\\Matlab\\D037_28_01_21a.edf'
id = 'D037'
raw = mne.io.read_raw_edf(edf)
sampling_rate = int(raw.info['sfreq'])
# raw.pick_channels(['RAH1M'])  # 402
# raw.pick_channels(['EEG C3-REF'])  # raw
# raw.pick_channels(['SEEG RAH1-REF'])  # 406
# raw.crop(tmin=0, tmax=300)


for channel in raw.info['ch_names']:
    detect(raw.copy().pick_channels([channel]), sampling_rate, plot=False)

# detect(data, sampling_rate, plot=False)
print('finish detect')
print(datetime.now() - start_time)
if save_csv:
    spikes_df = pd.DataFrame(spikes_counter, columns=['channel', 'amp', 'grad', 'env'])
    spikes_df.to_csv(id + '_count.csv', index=False)
