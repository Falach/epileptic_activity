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
# from mff_to_edf import write_edf
from mnelab.io.writers import write_edf



# epilepsy
def get_thresh_id(name):
    if name == 'amp':
        return 1
    elif name == 'grad':
        return 2
    else:
        return 3


def save_as_fif():
    raw = mne.io.read_raw_edf('/Users/rotemfalach/projects/epileptic_activity/results/D036/D036_15_01_21a.edf')
    write_edf(mne_raw=raw, fname='example2.edf', overwrite=True)

# save_as_fif()

def simple_view():
    # edf = '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/P402_staging_PSG_and_intracranial_Mref_correct.txt.edf'
    raw = mne.io.read_raw_edf('/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/better_sample_rate/P402_overnightData.edf')
    # raw.pick_channels(['EOG1', 'EOG2', 'EMG', 'C3', 'C4', 'Fz', 'Pz', 'RAH1M', 'LAH1M'])
    raw.pick_channels(['EOG EOG1-REF', 'EOG EOG2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG PZ-REF', 'SEEG RAH1-REF'])
    raw.crop(tmin=500, tmax=3000)
    # types = mne.io.pick.channel_indices_by_type(raw.info)
    raw.set_channel_types({'SEEG RAH1-REF': 'seeg', 'EOG EOG1-REF': 'eog', 'EOG EOG2-REF': 'eog'})
    # raw.set_channel_types({'RAH1M': 'seeg', 'LAH1M': 'seeg'})
    # raw.set_channel_types({x :'seeg' for x in ['RA1M', 'REC1M', 'RAH1M', 'RPSMA1M', 'ROF1M', 'RAC1M', 'LA1M', 'LEC1M', 'LAH1M', 'LPSMA1M', 'LOF1M', 'LAC1M']})
    # some_events = mne.make_fixed_length_events(raw, start=5, stop=50, duration=2.)
    # mapping = {1: 'amp', 2: 'grad', 3: 'env'}
    # spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/spikes.csv')
    # some_events = np.array([[x[4], 0, get_thresh_id(x[1])] for x in spikes_df.values])
    # annot_from_events = mne.annotations_from_events(events=some_events, event_desc=mapping, sfreq=raw.info['sfreq'])
    # raw.set_annotations(annot_from_events)
    info = mne.create_info(['FILTER'], raw.info['sfreq'], ['seeg'])
    channel_for_filter = mne.io.RawArray(raw.get_data()[5].reshape(1, raw.n_times), info)
    filtered_channel = mne.filter.filter_data(channel_for_filter, h_freq=200, l_freq=None, sfreq=2000, copy=True)
    raw.add_channels([filtered_channel], force_update_info=True)
    raw.plot(start=0, duration=30, scalings=dict(seeg=1e-3, eeg=1e-4))


    print('finish')


# simple_view()


def plot_with_events():
    edf = '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/402_for_tag.edf'
    raw = mne.io.read_raw_edf(edf)
    raw.pick_channels(['RAH1', 'RAH1-RAH2'])
    raw.crop(tmin=0, tmax=600)
    # raw.set_channel_types({'SEEG RAH1-REF': 'seeg', 'EOG1-REF': 'eog', 'EOG2-REF': 'eog'})
    # raw.set_channel_types({x :'seeg' for x in ['RA1M', 'REC1M', 'RAH1M', 'RPSMA1M', 'ROF1M', 'RAC1M', 'LA1M', 'LEC1M', 'LAH1M', 'LPSMA1M', 'LOF1M', 'LAC1M']})
    # some_events = mne.make_fixed_length_events(raw, start=5, stop=50, duration=2.)
    mapping = {1: 'amp', 2: 'grad', 3: 'env'}
    spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/402_RAH1_spikes.csv')
    some_events = np.array([[x[4], 0, get_thresh_id(x[1])] for x in spikes_df.values])
    annot_from_events = mne.annotations_from_events(events=some_events, event_desc=mapping, sfreq=raw.info['sfreq'])
    raw.set_annotations(annot_from_events)
    # raw.save('406_events.fif')
    raw.plot(color='black', start=0, duration=30, scalings=dict(eeg=1.2e-4, eog=3e-4, seeg=2e-4))
    print('finish')


# plot_with_events()


# def read_nicolet():
#     raw = mne.io.read_raw_nihon('/Users/rotemfalach/Documents/University/lab/Firas/D034/Patient52_LTM-1_t1.e')


# read_nicolet()
# good scale for 402
# raw.plot(start=0, duration=30, scalings=dict(eeg=55, seeg=180))

# fig = viz.plot_raw(raw, duration=30, scalings=dict(eeg=30, seeg=140))
# viz.plot_raw(raw, duration=30)



def spikes_index(data, sf, time, hypno):
    spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/results/406/406_RAH_spikes.csv')
    # spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/402_all_spikes.csv')
    index_list = [[(row['first_index'] - 3) / 4, (row['last_index'] + 4) / 4] for index, row in spikes_df.iterrows()]
    return np.array(index_list)


def peak_index(data, sf, time, hypno):
    spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/406_all_edf_spikes.csv')
    # spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/402_all_spikes.csv')
    index_list = [[row['max_index'] / 4, row['max_index'] / 4] for index, row in spikes_df.iterrows()]
    return np.array(index_list)


def visbrain_spikes_red():
    # edf = '/Users/rotemfalach/projects/python_eeg_analysis/406_events.edf'
    edf = '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/P402_staging_PSG_and_intracranial_Mref_correct.txt.edf'
    raw = mne.io.read_raw_edf(edf)
    # raw.pick_channels(['EOG1', 'EOG2', 'EMG', 'C3', 'C4', 'Fz', 'Pz', 'RAH1M', 'LAH1M'])
    # raw.crop(tmin=500, tmax=3000)
    # raw.set_channel_types({'RAH1M': 'seeg', 'LAH1M': 'seeg'})
    # data, sf, chan = raw.get_data(), raw.info['sfreq'], raw.info['ch_names']
    sp = Sleep(data=edf)
    sp.replace_detections('spindle', spikes_index)
    sp.show()


# visbrain_spikes_red()


def visbrain_spikes_point():
    edf = '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/better_sample_rate/P406_overnightData_filtered.edf'
    raw = mne.io.read_raw_edf(edf)
    raw.pick_channels(['SEEG RAH1-REF'])
    raw.crop(tmin=0, tmax=20000)
    raw.load_data()
    sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names']).show()

    sp.replace_detections('peak', peak_index)
    sp.show()


# visbrain_spikes_point()


def new_event_channel():
    edf = '/Users/rotemfalach/Documents/University/lab/EDFs_forRotem/better_sample_rate/P406_overnightData_filtered.edf'
    raw = mne.io.read_raw_edf(edf)
    raw.pick_channels(['EOG1-REF', 'EOG2-REF', 'C3-REF', 'C4-REF', 'PZ-REF', 'SEEG RAH1-REF'])
    raw.crop(tmin=0, tmax=1000)
    raw.set_channel_types({'SEEG RAH1-REF': 'seeg', 'EOG1-REF': 'eog', 'EOG2-REF': 'eog'})
    mapping = {1: 'amp', 2: 'grad', 3: 'env'}
    spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/406_raw_spikes.csv')
    some_events = np.array([[x[4], 0, get_thresh_id(x[1])] for x in spikes_df.values])
    stim_data = np.zeros((1, raw.n_times))
    for evt in some_events:
        stim_data[0][evt[0]] = evt[2] * 10
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.load_data()
    raw.add_channels([stim_raw], force_update_info=True)
    raw.reorder_channels(['EOG1-REF', 'EOG2-REF', 'C3-REF', 'C4-REF', 'PZ-REF', 'SEEG RAH1-REF', 'STI'])
    Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names']).show()

    # raw.plot(color='black', start=0, duration=30, scalings=dict(eeg=1.2e-4, eog=3e-4, seeg=2e-4))
    print('finish')


# new_event_channel()


def three_event_channel():
    edf = '/Users/rotemfalach/projects/epileptic_activity/results/406/P406_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    raw.pick_channels(['SEEG RAH1-REF'])
    # raw.pick_channels(['EOG1-REF', 'EOG2-REF', 'C3-REF', 'C4-REF', 'PZ-REF', 'SEEG RAH1-REF'])
    stim_amp, stim_grad, stim_env = np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times))
    spikes_df = pd.read_csv('/Users/rotemfalach/projects/epileptic_activity/results/406/406_RAH_spikes.csv')
    for evt in spikes_df.values:
        if evt[1] == 'amp':
            stim_amp[0][evt[4]] = 1
        elif evt[1] == 'grad':
            stim_grad[0][evt[4]] = 1
        else:
            stim_env[0][evt[4]] = 1

    info_amp = mne.create_info(['AMP'], raw.info['sfreq'], ['stim'])
    info_grad = mne.create_info(['GRAD'], raw.info['sfreq'], ['stim'])
    info_env = mne.create_info(['ENV'], raw.info['sfreq'], ['stim'])
    amp_raw = mne.io.RawArray(stim_amp, info_amp)
    grad_raw = mne.io.RawArray(stim_grad, info_grad)
    env_raw = mne.io.RawArray(stim_env, info_env)
    raw.load_data()
    raw.add_channels([amp_raw, grad_raw, env_raw], force_update_info=True)
    raw.reorder_channels(['SEEG RAH1-REF', 'AMP', 'GRAD', 'ENV'])
    raw.resample(500)
    sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names'], downsample=None)
    sp.replace_detections('peak', peak_index)
    sp.replace_detections('spindle', spikes_index)
    sp.show()


    # raw.plot(color='black', start=0, duration=30, scalings=dict(eeg=1.2e-4, eog=3e-4, seeg=2e-4))
    print('finish')


# three_event_channel()


def write_events_fif():
    edf = '/Users/rotemfalach/projects/epileptic_activity/results/406/406_overnightData.edf'
    channels_list = ['SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF', 'SEEG RA1-REF', 'SEEG RA2-REF',
                     'SEEG RAH1-REF', 'SEEG RAH2-REF']
    # channels_list = ['LA 01', 'LA 02']
    spikes_files = '_'.join(edf.split('_')[0:2])
    original_raw = mne.io.read_raw_edf(edf)
    original_raw.pick_channels(channels_list)
    original_raw.load_data()
    raw = original_raw
    for chan in channels_list:
        original_raw.rename_channels({chan: chan.replace('SEEG ', '').replace('-REF', '')})
        stim_amp, stim_grad, stim_env = np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times))
        chan_alias = chan.replace('SEEG ', '').replace('-REF', '')
        spikes_df = pd.read_csv(spikes_files + '_' + chan_alias + '_spikes.csv')
        for evt in spikes_df.values:
            if evt[1] == 'amp':
                stim_amp[0][evt[4]] = 1
            elif evt[1] == 'grad':
                stim_grad[0][evt[4]] = 1
            else:
                stim_env[0][evt[4]] = 1

        info_amp = mne.create_info([chan_alias + '_AMP'], raw.info['sfreq'], ['stim'])
        info_grad = mne.create_info([chan_alias + '_GRAD'], raw.info['sfreq'], ['stim'])
        info_env = mne.create_info([chan_alias + '_ENV'], raw.info['sfreq'], ['stim'])
        amp_raw = mne.io.RawArray(stim_amp, info_amp)
        grad_raw = mne.io.RawArray(stim_grad, info_grad)
        env_raw = mne.io.RawArray(stim_env, info_env)
        original_raw.add_channels([amp_raw, grad_raw, env_raw], force_update_info=True)

    original_raw.resample(500).save('406_regular.fif')


def write_events_fif_bipolar():
    edf = '/Users/rotemfalach/projects/epileptic_activity/results/406/406_overnightData.edf'
    channels_list = ['SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF', 'SEEG RA1-REF', 'SEEG RA2-REF',
                     'SEEG RAH1-REF', 'SEEG RAH2-REF']
    channels_list_bipolar = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list]
    # channels_list = ['LA 02', 'LH 01', 'LH 02', 'RA 02', 'RMH 01', 'RMH 02'] # 036
    # channels_list_bipolar = [x[:-1] + str(int(x[-1]) + 1) for x in channels_list]
    spikes_files = '_'.join(edf.split('_')[0:2])
    original_raw = mne.io.read_raw_edf(edf)
    original_raw.pick_channels(list(set(channels_list) | set(channels_list_bipolar)))
    original_raw.load_data()
    raw = original_raw
    for chan in channels_list:
        # original_raw.rename_channels({chan: chan.replace('SEEG ', '').replace('-REF', '')})
        original_raw = mne.set_bipolar_reference(original_raw, anode=[chan], cathode=chan[:-5] + str(int(chan[-5]) + 1) + '-REF',
                                                 drop_refs=False, ch_name=chan + '_bi')
        stim_amp, stim_grad, stim_env = np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times))
        # chan_alias = chan.replace(' 0', '')
        chan_alias = chan.replace('SEEG ', '').replace('-REF', '')
        spikes_df = pd.read_csv(spikes_files + '_' + chan_alias + '_biplor_spikes.csv')
        for evt in spikes_df.values:
            if evt[1] == 'amp':
                stim_amp[0][evt[4]] = 1
            elif evt[1] == 'grad':
                stim_grad[0][evt[4]] = 1
            else:
                stim_env[0][evt[4]] = 1

        info_amp = mne.create_info([chan_alias + '_AMP_bi'], raw.info['sfreq'], ['stim'])
        info_grad = mne.create_info([chan_alias + '_GRAD_bi'], raw.info['sfreq'], ['stim'])
        info_env = mne.create_info([chan_alias + '_ENV_bi'], raw.info['sfreq'], ['stim'])
        amp_raw = mne.io.RawArray(stim_amp, info_amp)
        grad_raw = mne.io.RawArray(stim_grad, info_grad)
        env_raw = mne.io.RawArray(stim_env, info_env)
        original_raw.add_channels([amp_raw, grad_raw, env_raw], force_update_info=True)

    original_raw.drop_channels(list(set(channels_list) | set(channels_list_bipolar)))
    original_raw.resample(500).save('406_bi.fif')


def find_overlap_spikes():
    rotem_RAH1_tag_sec = [41.2, 61.9, 87.6, 97.3, 121.5, 124, 128.3, 148.5, 150.2, 158.2, 175.7, 177.1, 186.5, 207.1,
                          220.5, 239.5, 280.6, 283.1]
    spikes_df = pd.read_csv('../epileptic_activity/402_RAH1_spikes.csv')
    overlap = []
    for x in rotem_RAH1_tag_sec:
        for i, y in spikes_df.iterrows():
            if y['first_index'] / 2000 - 0.4 < x < y['last_index'] / 2000 + 0.4:
                overlap.append(y)

    print(overlap)
    return 1

# find_overlap_spikes()


def intersect_spikes_union():
    spikes_df = pd.read_csv('../epileptic_activity/402_RAH1_spikes.csv')
    new_spikes_list = []
    for i, x in spikes_df.iterrows():
        for j, y in spikes_df[i + 1:].iterrows():
            # four cases- included in another spike, include anoter spike (redundent?), intersect from left and from right
            if (x['first_index'] > y['first_index'] and x['last_index'] < y['last_index']) or \
                (x['first_index'] < y['first_index'] and x['last_index'] > y['last_index']) or \
                (x['first_index'] < y['last_index'] < x['last_index']) or \
                (x['first_index'] < y['first_index'] < x['last_index']):
                    max_index, max_amp = (x['max_index'], x['max_amp']) if x['max_amp'] > y['max_amp'] else (y['max_index'], y['max_amp'])
                    thresh_type = x['threshold_type'] if x['threshold_type'] == y['threshold_type'] else [x['threshold_type'], y['threshold_type']]
                    new_spikes_list.append([thresh_type, min(x['first_index'], y['first_index']), max(x['last_index'], y['last_index']),
                                            max_index, max_amp, abs(x['first_index']-y['first_index'])])

    new_spikes_df = pd.DataFrame(new_spikes_list,
                             columns=['threshold_type', 'first_index', 'last_index', 'max_index', 'max_amp', 'start_diff'])
    new_spikes_df = new_spikes_df.drop_duplicates(subset='max_index')
    new_spikes_df['duration'] = spikes_df['last_index'] - spikes_df['first_index']
    new_spikes_df.to_csv('402_RAH1_union.csv', index=False)


intersect_spikes_union()


def intersect_spikes_union_by_50():
    spikes_df = pd.read_csv('../epileptic_activity/402_RAH1_spikes.csv')
    new_spikes_list = []
    for i, x in spikes_df.iterrows():
        for j, y in spikes_df[i + 1:].iterrows():
            # 0.05 sec * sample rate is 100
            if abs(x['max_index'] - y['max_index']) < 100:
                    max_index, max_amp = (x['max_index'], x['max_amp']) if x['max_amp'] > y['max_amp'] else (y['max_index'], y['max_amp'])
                    thresh_type = x['threshold_type'] if x['threshold_type'] == y['threshold_type'] else [x['threshold_type'], y['threshold_type']]
                    new_spikes_list.append([thresh_type, min(x['first_index'], y['first_index']), max(x['last_index'], y['last_index']),
                                            max_index, max_amp, abs(x['first_index']-y['first_index'])])

    new_spikes_df = pd.DataFrame(new_spikes_list,
                             columns=['threshold_type', 'first_index', 'last_index', 'max_index', 'max_amp', 'start_diff'])
    new_spikes_df = new_spikes_df.drop_duplicates(subset='max_index')
    new_spikes_df['duration'] = spikes_df['last_index'] - spikes_df['first_index']
    new_spikes_df.to_csv('402_RAH1_union_by_50.csv', index=False)


# intersect_spikes_union_by_50()