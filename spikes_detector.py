import mne
from mne import viz
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats, signal

# general constants
sampling_rate = 1000
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
percentageOfNansAllowedArounsSpike = 0.1  # how many NaNs are allowed in the vicinity of the spike(vicinity=minDistSpikes / 2 before and after)
HighBandPassScore = 11  # this is for debugging - finding especially high STDs for HP


def get_markers(data, curr_block, points_above_thresh, z_score):
    max_markers_index = []
    max_marker_value = []

    if len(points_above_thresh) > 0:
        index_above_threshold = (z_score > threshold_amp).nonzero()[0]

        # find max markers
        counter = 0
        for j in range(len(index_above_threshold)):
            # check that the next index is the same spike
            if j + 1 < len(index_above_threshold) and index_above_threshold[j + 1] - index_above_threshold[j] == 1:
                counter += 1
            # the current spike finished
            else:
                if counter <= 1:
                    value = curr_block[index_above_threshold[j]]
                else:
                    value = curr_block[index_above_threshold[j - counter]: index_above_threshold[j] + 1].max()
                max_marker_value.append(value)
                max_markers_index.append(np.where(data == value)[0][0])
                counter = 1

    return max_markers_index, max_marker_value


def detect(data):
    points_in_block = block_size_sec * sampling_rate
    number_of_blocks = math.floor(len(data) / points_in_block)

    plt.plot(data, alpha=0.8)

    for i in range(number_of_blocks):
        curr_block = data[i * points_in_block: (i + 1) * points_in_block]

        # check amplitude threshold
        if use_amp or use_amp_env or use_amp_grad:
            z_score_amp = stats.zscore(curr_block)
            points_above_thresh_amp = z_score_amp[z_score_amp > threshold_amp]

        # check gradient threshold
        if use_grad or use_amp_grad:
            gradient_diff = np.diff(curr_block)
            z_score_grad = stats.zscore(np.insert(gradient_diff, 0, 0))
            points_above_thresh_grad = z_score_grad[z_score_grad > threshold_grad]

        # check envelope threshold
        if use_env or use_amp_env:
            filtered_block = mne.filter.filter_data(curr_block, sampling_rate, low_pass, high_pass)
            env_block = abs(signal.hilbert(filtered_block))
            z_score_env = stats.zscore(env_block)
            points_above_thresh_env = z_score_env[z_score_env > threshold_env]

        # combine thresholds and get spikes points
        max_markers_index_amp, max_marker_value_amp = get_markers(data, curr_block, points_above_thresh_amp, z_score_amp)
        max_markers_index_grad, max_marker_value_grad = get_markers(data, curr_block, points_above_thresh_grad, z_score_grad)
        max_markers_index_env, max_marker_value_env = get_markers(data, curr_block, points_above_thresh_env, z_score_env)

        common_index = np.intersect1d(max_markers_index_amp, max_markers_index_grad)
        all_common_index = np.intersect1d(common_index, max_markers_index_env)
        all_common_value = data[all_common_index] if len(common_index) > 0 else []

        plt.scatter(all_common_index, all_common_value, marker='D', color='green')
        plt.scatter(np.setdiff1d(max_markers_index_amp, all_common_index), np.setdiff1d(max_marker_value_amp, all_common_value), marker='X', color='black')
        plt.scatter(np.setdiff1d(max_markers_index_grad, all_common_index), np.setdiff1d(max_marker_value_grad, all_common_value), marker='P', color='red')
        plt.scatter(np.setdiff1d(max_markers_index_env, all_common_index), np.setdiff1d(max_marker_value_env, all_common_value), marker='o', color='blue')
        # plt.scatter(max_markers_index_env, max_marker_value_env, marker='X', color='black')

    print('cool')

    return True


raw = mne.io.read_raw_edf(
    '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/P402_staging_PSG_and_intracranial_Mref_correct.txt.edf')
raw.pick_channels(['RAH1M'])
raw.crop(tmin=1900, tmax=2000)

# viz.plot_raw(raw, duration=30, scalings=dict(eeg=35))
# plt.plot(raw.get_data()[0])

data = raw.get_data()[0]
detect(data)

