import pyedflib
from datetime import datetime, timezone, timedelta
import mne
from mne import viz
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
from visbrain.gui import Sleep
from scipy.signal import detrend
from subjects_config import *

# choose a patient and a channel
# 406, 22:40-5:15
# selected_patient = p_406
# selected_channel = 'SEEG RAH1-REF'
# spikes_df = pd.read_csv(selected_patient['spikes_RAH1'])
# selected_channel = 'SEEG RPHG1-REF'
# spikes_df = pd.read_csv(selected_patient['spikes_RPHG1'])

# D036, 18:00-9:20
selected_patient = p_D036
selected_channel = 'LA 02'
# selected_channel = 'LPHGp 04'
# selected_channel = 'RA 08'
# spikes_df = pd.read_csv(selected_patient['spikes_' + selected_channel.replace(' ', '_')])
spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/results/D036/D036_LA2_spikes.csv')


# D037, 15:30-9:00
# selected_patient = p_D037
# selected_channel = 'LAH 01'
# selected_channel = 'RH 01'
# spikes_df = pd.read_csv(selected_patient['spikes_' + selected_channel.replace(' ', '_')])


raw = mne.io.read_raw_edf(selected_patient['edf'])
original_sf = int(raw.info['sfreq'])


def spikes_index(data, sf, time, hypno):
    if original_sf == 2000:
        index_list = [[(row['first_index'] - 3) / 4, (row['last_index'] + 4) / 4] for index, row in
                      spikes_df.iterrows()]
    else:
        index_list = [[row['first_index'] - 3, row['last_index'] + 4] for index, row in
                      spikes_df.iterrows()]

    return np.array(index_list)


def peak_index(data, sf, time, hypno):
    if original_sf == 2000:
        index_list = [[row['max_index'] / 4, row['max_index'] / 4] for index, row in spikes_df.iterrows()]
    else:
        index_list = [[row['max_index'] / 4, row['max_index'] / 4] for index, row in spikes_df.iterrows()]
    return np.array(index_list)


def three_event_channel():
    stim_amp, stim_grad, stim_env = np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times))
    for evt in spikes_df.values:
        if evt[1] == 'amp':
            stim_amp[0][evt[4]] = 1
        elif evt[1] == 'grad':
            stim_grad[0][evt[4]] = 1
        else:
            stim_env[0][evt[4]] = 1

    raw.pick_channels([selected_channel])
    info_amp = mne.create_info(['AMP'], raw.info['sfreq'], ['stim'])
    info_grad = mne.create_info(['GRAD'], raw.info['sfreq'], ['stim'])
    info_env = mne.create_info(['ENV'], raw.info['sfreq'], ['stim'])
    amp_raw = mne.io.RawArray(stim_amp, info_amp)
    grad_raw = mne.io.RawArray(stim_grad, info_grad)
    env_raw = mne.io.RawArray(stim_env, info_env)
    raw.load_data()
    raw.add_channels([amp_raw, grad_raw, env_raw], force_update_info=True)
    raw.reorder_channels([selected_channel, 'AMP', 'GRAD', 'ENV'])
    if original_sf > 1000:
        raw.resample(500)
    sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names'], downsample=None)
    sp.replace_detections('peak', peak_index)
    sp.replace_detections('spindle', spikes_index)
    sp.show()
    print('finish')


three_event_channel()
