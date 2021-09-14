import mne
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
from scipy import stats, signal
import itertools
import pandas as pd
import utils
from datetime import datetime
# mpl.use('Agg')
mne.set_log_level(False)

# general constants
min_distance = 50  # minimal distance for 'different' spikes - in miliseconds

# constants for detection based on frequency analysis - the thresholds for the standard deviation are based on the
# papers Andrillon et al(envelope condition) and Starestina et al(other conditions)
threshold_env = 8  # threshold in standard deviations for the envelope after bandpass (HP)
threshold_amp = 5  # threshold in standard deviations for the amplitude
threshold_grad = 5  # threshold in standard deviations for the gradient
use_env = True
use_amp = True
use_grad = True
use_amp_grad = True
use_amp_env = True
block_size_sec = 10  # filter and find peaks at blocks of X seconds - based on Andrillon et al

# the bandpass range is based on Andrillon et al
low_pass = 50
high_pass = 150

min_spike_length_ms = 5  # a spike is detected if there are points for X ms passing the threshold - in ms, based on Andrillon et al
common_spikes_index = []

# format: [spike_id, threshold_type, first_index, last_index, max_index, max_amp, duration(calc once for all after convert to df), sleep_stage?]
spikes_list = []


def get_markers(data, index_above_threshold, thresh_type, sr):
    max_markers_index = []
    max_marker_value = []
    min_spike_points = sr / (1000 / min_spike_length_ms)
    # find max markers
    counter = 1
    curr_spike = [thresh_type, index_above_threshold[0]]
    for j in range(len(index_above_threshold)):
        # check that the next index is the same spike
        if j + 1 < len(index_above_threshold) and index_above_threshold[j + 1] - index_above_threshold[j] == 1:
            counter += 1
        # the current spike finished
        else:
            # check min time of spike
            if counter >= min_spike_points:
                # check if the peak is positive or negative and append it's value
                max_value = data[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].max()
                min_value = data[index_above_threshold[j - counter + 1]: index_above_threshold[j] + 1].min()
                value = max_value if abs(max_value) > abs(min_value) else min_value
                index = np.intersect1d(np.where(data == value)[0], index_above_threshold[j - counter + 1: j + 1])[0]
                max_marker_value.append(value)
                max_markers_index.append(index)
                curr_spike.extend((index_above_threshold[j], index, value))
                spikes_list.append(curr_spike)

            if j + 1 < len(index_above_threshold):
                curr_spike = [thresh_type, index_above_threshold[j + 1]]
                counter = 1

    return np.array(max_markers_index), np.array(max_marker_value)


def detect(data, sampling_rate, thresh_amp, thresh_grad, thresh_env, plot=True):
    points_in_block = block_size_sec * sampling_rate
    number_of_blocks = math.floor(len(data) / points_in_block)
    max_markers_index_amp, max_markers_index_grad, max_markers_index_env, all_common_index, all_common_value = [], [], [], [], []
    if plot:
        plt.plot(data, alpha=0.8)

    for i in range(number_of_blocks):
        curr_block = data[i * points_in_block: (i + 1) * points_in_block]

        # check amplitude threshold
        if use_amp or use_amp_env or use_amp_grad:
            z_score_amp = stats.zscore(curr_block)
            points_above_thresh_amp = z_score_amp[z_score_amp > thresh_amp]
            # get indexes from z_score values and add offset of the current block
            if len(points_above_thresh_amp) > 0:
                index_above_threshold_amp = (z_score_amp > thresh_amp).nonzero()[0] + i * points_in_block
                max_markers_index_amp, max_marker_value_amp = get_markers(data, index_above_threshold_amp, 'amp', sampling_rate)

        # check gradient threshold
        if use_grad or use_amp_grad:
            gradient_diff = np.diff(curr_block)
            z_score_grad = stats.zscore(np.insert(gradient_diff, 0, 0))
            points_above_thresh_grad = z_score_grad[z_score_grad > thresh_grad]
            if len(points_above_thresh_grad) > 0:
                index_above_threshold_grad = (z_score_grad > thresh_grad).nonzero()[0] + i * points_in_block
                max_markers_index_grad, max_marker_value_grad = get_markers(data, index_above_threshold_grad, 'grad', sampling_rate)

        # check envelope threshold
        if use_env or use_amp_env:
            filtered_block = mne.filter.filter_data(curr_block, sampling_rate, low_pass, high_pass)
            env_block = abs(signal.hilbert(filtered_block))
            z_score_env = stats.zscore(env_block)
            points_above_thresh_env = z_score_env[z_score_env > thresh_env]
            if len(points_above_thresh_env) > 0:
                index_above_threshold_env = (z_score_env > thresh_env).nonzero()[0] + i * points_in_block
                max_markers_index_env, max_marker_value_env = get_markers(data, index_above_threshold_env, 'env', sampling_rate)

        if plot:
            # find the points that are shared in all thresholds
            common_index = np.intersect1d(max_markers_index_amp, max_markers_index_grad)
            if len(common_index) > 0:
                all_common_index = np.intersect1d(common_index, max_markers_index_env)
                common_spikes_index.extend(all_common_index)
                all_common_value = data[all_common_index] if len(all_common_index) > 0 else []

                # remove the shared points
                max_markers_index_amp = max_markers_index_amp[~np.in1d(max_markers_index_amp, all_common_index)]
                max_markers_index_grad = max_markers_index_grad[~np.in1d(max_markers_index_grad, all_common_index)]
                max_markers_index_env = max_markers_index_env[~np.in1d(max_markers_index_env, all_common_index)]

            # draw
            plt.scatter(all_common_index, all_common_value, marker='D', color='black')
            plt.scatter(max_markers_index_amp, data[max_markers_index_amp] if len(max_markers_index_amp) > 0 else [], marker='X', color='fuchsia')
            plt.scatter(max_markers_index_grad, data[max_markers_index_grad] if len(max_markers_index_grad) > 0 else [], marker='P', color='red')
            plt.scatter(max_markers_index_env, data[max_markers_index_env] if len(max_markers_index_env) > 0 else [], marker='o', color='blue', s=15)
            plt.legend(['signal', 'all', 'amplitude', 'gradient', 'envelope'], loc='upper right')
            # plt.scatter(max_markers_index_env, max_marker_value_env, marker='X', color='black')

    print('cool')
    if plot:
        plt.close()

    return True


# run all start
print(datetime.now())
subjects_edf = [
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\396\\396_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\398\\398_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\402\\402_for_tag.edf',
                # 'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\405\\405_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\406\\406_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\415\\415_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\416\\416_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\34\\34_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\36\\36_for_tag.edf',
                'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\37\\37_for_tag.edf'
                ]
for subj in subjects_edf:
    raw = mne.io.read_raw_edf(subj)
    sampling_rate = int(raw.info['sfreq'])
    for chan in [x for x in raw.ch_names if 'RA' in x or 'LA' in x]:
        chan_data = raw.copy().pick_channels([chan]).get_data()[0]
        for block_size in [3, 5, 10]:
            block_size_sec = block_size
            for amp_thresh in [5, 6, 7, 8, 9, 10]:
                for grad_thresh in [5, 6, 7, 8, 9, 10]:
                    for env_thresh in [5, 6, 7, 8, 9, 10]:
                        if not amp_thresh == grad_thresh == env_thresh:
                            detect(chan_data, sampling_rate, amp_thresh, grad_thresh, env_thresh, False)
                            spikes_df = pd.DataFrame(spikes_list,
                                                     columns=['threshold_type', 'first_index', 'last_index', 'max_index', 'max_amp'])
                            # spikes_df['duration'] = spikes_df['last_index'] - spikes_df['first_index']
                            union_spikes = utils.union_spikes(spikes_df, min_distance, sampling_rate)
                            union_spikes.to_csv(subj.replace('.edf', '') + f'_{chan}_b{block_size}_t{amp_thresh}_{grad_thresh}_{env_thresh}.csv')
                            spikes_list = []


        print(f'finish channel {chan}')
        print(datetime.now())

    print(f'finish subj {subj}')
    print(datetime.now())


