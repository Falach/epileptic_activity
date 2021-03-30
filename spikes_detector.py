import mne
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats, signal

# general constants
# TODO: check how to handle 128 Hz
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
percentageOfNansAllowedAroundSpike = 0.1  # how many NaNs are allowed in the vicinity of the spike(vicinity=minDistSpikes / 2 before and after)
HighBandPassScore = 11  # this is for debugging - finding especially high STDs for HP


def get_markers(curr_block, points_above_thresh, z_score):
    max_markers_index = []
    max_marker_value = []

    # get indexes from z_score values
    if len(points_above_thresh) > 0:
        index_above_threshold = (z_score > threshold_amp).nonzero()[0]

        # find max markers
        counter = 1
        for j in range(len(index_above_threshold)):
            # check that the next index is the same spike
            if j + 1 < len(index_above_threshold) and index_above_threshold[j + 1] - index_above_threshold[j] == 1:
                counter += 1
            # the current spike finished
            else:
                if counter == 1:
                    max_marker_value.append(curr_block[index_above_threshold[j]])
                    max_markers_index.append(index_above_threshold[j])
                else:
                    # check if the peak is positive or negative and append it's value
                    max_value = curr_block[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].max()
                    min_value = curr_block[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].min()
                    value = max_value if abs(max_value) > abs(min_value) else min_value
                    max_marker_value.append(value)
                    max_markers_index.append(np.intersect1d(np.where(curr_block == value)[0], index_above_threshold[j - counter + 1: j + 1])[0])
                    counter = 1

    return np.array(max_markers_index), np.array(max_marker_value)


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
        max_markers_index_amp, max_marker_value_amp = get_markers(curr_block, points_above_thresh_amp, z_score_amp)
        max_markers_index_grad, max_marker_value_grad = get_markers(curr_block, points_above_thresh_grad, z_score_grad)
        max_markers_index_env, max_marker_value_env = get_markers(curr_block, points_above_thresh_env, z_score_env)

        # set indexes to the total data
        max_markers_index_amp = np.array(max_markers_index_amp) + i * points_in_block
        max_markers_index_grad = np.array(max_markers_index_grad) + i * points_in_block
        max_markers_index_env = np.array(max_markers_index_env) + i * points_in_block

        # find the points that are shared in all thresholds
        common_index = np.intersect1d(max_markers_index_amp, max_markers_index_grad)
        all_common_index = np.intersect1d(common_index, max_markers_index_env)
        all_common_value = data[all_common_index] if len(all_common_index) > 0 else []
        plt.scatter(all_common_index, all_common_value, marker='D', color='black')

        # remove the shared points
        max_markers_index_amp = max_markers_index_amp[~np.in1d(max_markers_index_amp, all_common_index)]
        max_markers_index_grad = max_markers_index_grad[~np.in1d(max_markers_index_grad, all_common_index)]
        max_markers_index_env = max_markers_index_env[~np.in1d(max_markers_index_env, all_common_index)]

        # draw
        plt.scatter(max_markers_index_amp, data[max_markers_index_amp] if len(max_markers_index_amp) > 0 else [], marker='X', color='fuchsia')
        plt.scatter(max_markers_index_grad, data[max_markers_index_grad] if len(max_markers_index_grad) > 0 else [], marker='P', color='red')
        plt.scatter(max_markers_index_env, data[max_markers_index_env] if len(max_markers_index_env) > 0 else [], marker='o', color='blue', s=15)
        plt.legend(['signal', 'all', 'amplitude', 'gradient', 'envelope'])
        # plt.scatter(max_markers_index_env, max_marker_value_env, marker='X', color='black')

    print('cool')

    return plt


raw = mne.io.read_raw_edf(
    '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/P402_staging_PSG_and_intracranial_Mref_correct.txt.edf')
raw.pick_channels(['RAH1M'])
raw.crop(tmin=1000, tmax=7000)

# viz.plot_raw(raw, duration=30, scalings=dict(eeg=35))
# plt.plot(raw.get_data()[0])

data = raw.get_data()[0]
detect(data)

