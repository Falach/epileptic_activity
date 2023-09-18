import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from visbrain.gui import Sleep
# from mff_to_edf import write_edf
from mnelab.io.writers import write_edf
import h5py
import glob
import scipy.io as sio
from mff_to_edf import write_edf as rotem_write_edf
from mnelab.io.writers import write_edf as mnelab_edf
from scipy.interpolate import make_interp_spline
from scipy.stats import kde
import seaborn as sns



# epilepsy
def get_thresh_id(name):
    if name == 'amp':
        return 1
    elif name == 'grad':
        return 2
    else:
        return 3


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


# intersect_spikes_union()


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

# union close spikes
def union_spikes(spikes_df, min_distance_sec, sr):
    new_spikes_list = []
    flag, any_union = False, False
    for i, x in spikes_df.iterrows():
        # prevent double unions
        for j, y in spikes_df[i + 1:].iterrows():
            # check distance between the peaks
            if abs(x['max_index'] - y['max_index']) / sr < min_distance_sec:
                # more than one union with the current spike x
                if flag:
                    thresh_type = list(set(curr_union[0] + [y['threshold_type']]))
                    max_index, max_amp = (curr_union[3], curr_union[4]) if abs(curr_union[4]) > abs(y['max_amp']) \
                        else (y['max_index'], y['max_amp'])
                    curr_union = [thresh_type, min(curr_union[1], y['first_index']), max(curr_union[2], y['last_index']),
                         max_index, max_amp, abs(curr_union[1] - y['first_index'])]
                else:
                    thresh_type = list(set([x['threshold_type'], y['threshold_type']]))
                    max_index, max_amp = (x['max_index'], x['max_amp']) if abs(x['max_amp']) > abs(y['max_amp']) \
                            else (y['max_index'], y['max_amp'])
                    curr_union = [thresh_type, min(x['first_index'], y['first_index']), max(x['last_index'], y['last_index']),
                         max_index, max_amp, abs(x['first_index'] - y['first_index'])]
                flag, any_union = True, True
        if flag:
            new_spikes_list.append(curr_union)
            flag = False
        else:
            new_spikes_list.append(x.tolist())

    columns = ['threshold_type', 'first_index', 'last_index', 'max_index', 'max_amp', 'start_diff']
    if not any_union:
        columns.remove('start_diff')
    new_spikes_df = pd.DataFrame(new_spikes_list, columns=columns)
    new_spikes_df = new_spikes_df.drop_duplicates(subset='max_index')
    new_spikes_df['duration'] = spikes_df['last_index'] - spikes_df['first_index']
    return new_spikes_df

def crop_for_tag():
    # this is the code that crop data for tagging
    ids = ['34', '36', '37', '396', '398', '402', '405', '406', '415', '416']
    start_in_minutes = [211, 186, 393, 18, 10, 9, 6, 116, 59, 132]

    for id, start_time in zip(ids, start_in_minutes):
        getattr(save_biploar, f'for_firas_{id}')(start_time, f'{id}_for_tag_250hz.edf')

def preprocess():
    # save raw files after apply filter and notch
    for subj in ['396', '398', '402', '405', '406', '415', '416']:
        raw = mne.io.read_raw_edf(f'C:\\UCLA\\{subj}_cz+bi_full.edf')
        raw.load_data()
        raw.filter(l_freq=0.1, h_freq=500)
        raw.notch_filter((60, 120, 180, 240), method='spectrum_fit')
        raw.save(f'C:\\UCLA\\{subj}_cz+bi_full_filtered.fif')
        write_edf(f'C:\\UCLA\\{subj}_cz+bi_full_filtered.edf', raw)

def from_nicolet_to_mat_to_edf():
    from mff_to_edf import write_edf as rotem_write_edf

    subj = '39'
    f = h5py.File(f'D:\\Firas\\D0{subj}\\D0{subj}.mat', 'r')
    data = f.get('dat')
    data = np.array(data)
    # ch_names = mne.io.read_raw_edf('C:\\Matlab\\D037_04_02_21b.edf').info['ch_names']
    ch_names = pd.read_csv(f'D:\\Firas\\D0{subj}\\D0{subj}_chans.csv', header=None)
    ch_names = [x.replace('\'', '') for x in ch_names.iloc[:, 0].tolist()]
    sfreq = int(np.array(f.get('hdr/Fs'))[0][0])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
    mne_raw = mne.io.RawArray(data.T, info)
    # mne_raw.plot()
    scalp_chans = ['Pz', 'Fz', 'Cz', 'EOG1', 'EOG2']
    depth_chans = ['RAH1', 'RAH2', 'LAH1', 'LAH2', 'LA1']
    # mne_raw.pick_channels(scalp_chans + depth_chans)
    # mne_raw.load_data()
    # zero phase prevent delay
    # mne_raw.filter(l_freq=0.1, h_freq=250, picks=depth_chans, phase='zero-double')
    # mne_raw.notch_filter((50, 100, 150, 200), picks=scalp_chans + depth_chans, method='spectrum_fit',
    #                      phase='zero-double')
    # mne_raw.filter(l_freq=0.1, h_freq=40, picks=scalp_chans, phase='zero-double')
    rotem_write_edf(mne_raw, f'C:\\UCLA\\P{subj}_overnightData.edf')
    print()

# from_nicolet_to_mat_to_edf()

def from_mat_to_edf(subj):
    from mff_to_edf import write_edf as rotem_write_edf

    mne_raw = None
    subj_files_list = glob.glob(f'D:\\Maya\\p{subj}\\MACRO\\*')
    for curr_file in subj_files_list:
        try:
            f = h5py.File(curr_file, 'r')
            data = f.get('data')
            data = np.array(data)
            ch_name_array = np.array([x for x in f['LocalHeader/origName']], dtype='uint16')
            ch_name = ''
            for x in ch_name_array:
                ch_name += chr(x[0])
            sfreq = np.array(f['LocalHeader/samplingRate'])[0][0]
            info = mne.create_info(ch_names=[ch_name], sfreq=sfreq)
            if mne_raw is None:
                mne_raw = mne.io.RawArray(data.T, info)
            else:
                mne_raw.add_channels([mne.io.RawArray(data.T, info)])
        except OSError:
            pass
    # mne_raw.crop(tmin=76 * 60, tmax=80 * 60)
    mne_raw.load_data()
    rotem_write_edf(mne_raw, f'P{subj}_sample_for_tag.edf')
    # mne_raw.save(f'C:\\Maya\\p{subj}\\P{subj}.fif')
    print()


def from_mat_to_edf_Hanna(subj='017', scalp=None):
    mne_raw = None
    subj_files_list = glob.glob(f'D:\\Hanna\\D{subj}\\Macro\\*')
    sfreq = 1000
    ch_names = pd.read_csv(f'D:\\Hanna\\D{subj}\\montage.csv')
    macro_chans = ch_names[ch_names['IS_MACRO'] == 1]
    scalp_chans = ch_names[ch_names['IS_SCALP'] == 1]['LOCATION'].tolist()
    exclude_chans = ch_names[ch_names['EXCLUDE'] == 1]['LOCATION'].tolist()
    for i, curr_file in enumerate(subj_files_list):
        try:
            f = h5py.File(curr_file, 'r')
            data = f.get('denoised_data')
            data = np.array(data)
            chan_num = int(curr_file.split('_')[1].replace('.mat', ''))
            curr_name = macro_chans[macro_chans['CHANNEL'] == chan_num]['LOCATION'].tolist()[0]
             # = macro_chans['LOCATION'][chan_num].replace(' ', '')
            if curr_name not in exclude_chans:
                info = mne.create_info(ch_names=[curr_name], sfreq=sfreq)
                if mne_raw is None:
                    mne_raw = mne.io.RawArray(data.T, info)
                else:
                    if data.size > mne_raw.n_times:
                        data = data[:mne_raw.n_times]
                    mne_raw.add_channels([mne.io.RawArray(data.T, info)])
        except Exception as e:
            pass


    mne_raw.reorder_channels(sorted([x for x in mne_raw.ch_names if x not in scalp_chans]) + sorted(scalp_chans))
    # if scalp is not None:
    #     scalp = mne.io.read_raw_edf(f'D:\\Hanna\\D{subj}\\d{subj}_scalp_250Hz.edf').resample(sfreq)
    #     scalp.load_data()
    #     scalp_data = scalp.pick_channels(['Signal 0', 'Signal 1', 'Signal 2', 'Signal 3', 'Signal 4']).get_data()
    #     mne_raw.load_data()
    #     if mne_raw.n_times < scalp_data.shape[1]:
    #         mne_raw.add_channels([mne.io.RawArray(scalp_data[:, :mne_raw.n_times], scalp.info)], force_update_info=True)
    #     else:
    #         mne_raw.crop(tmax=scalp_data.shape[1] / sfreq - 1)
    #         mne_raw.add_channels([mne.io.RawArray(scalp_data[:, :mne_raw.n_times], scalp.info)], force_update_info=True)

    # rotem_write_edf(mne_raw, f'D:\\Hanna\\D{subj}\\P{subj}_overnightData_fixed.edf')
    write_edf(f'D:\\Hanna\\D{subj}\\P{subj}_overnightData_fixed_2.edf', mne_raw)
    print()

from_mat_to_edf_Hanna(subj='025')

# for subj in ['013', '017', '025']:
#     from_mat_to_edf_Hanna(subj=subj)

def from_mat_to_edf_Hanna_sio(subj='479'):
    mne_raw = None
    subj_files_list = glob.glob(f'D:\\Hanna\\D{subj}\\Macro\\*')
    sfreq = 1000
    ch_names = pd.read_csv(f'D:\\Hanna\\D{subj}\\montage.csv')
    macro_chans = ch_names[ch_names['IS_MACRO'] == 1]
    scalp_chans = ch_names[ch_names['IS_SCALP'] == 1]['LOCATION'].tolist()
    exclude_chans = ch_names[ch_names['EXCLUDE'] == 1]['LOCATION'].tolist()
    for i, curr_file in enumerate(subj_files_list):
        try:
            f = sio.loadmat(curr_file)
            data = f['denoised_data']
            data = np.array(data).T
            chan_num = int(curr_file.split('_')[1].replace('.mat', ''))
            curr_name = macro_chans[macro_chans['CHANNEL'] == chan_num]['LOCATION'].tolist()[0]
            if curr_name not in exclude_chans:
                info = mne.create_info(ch_names=[curr_name], sfreq=sfreq)
                if mne_raw is None:
                    mne_raw = mne.io.RawArray(data, info)
                else:
                    if data.size > mne_raw.n_times:
                        data = data[:mne_raw.n_times]
                    mne_raw.add_channels([mne.io.RawArray(data, info)])
        except Exception as e:
            pass

    # mne_raw.load_data()
    # scalp = mne.io.read_raw_edf(f'D:\\Hanna\\D{subj}\\d{subj}_scalp_250Hz.edf').resample(sfreq)
    # scalp.load_data()
    # scalp_data = scalp.pick_channels(['Signal 0', 'Signal 1', 'Signal 3', 'Signal 4']).get_data()
    # mne_raw.add_channels([mne.io.RawArray(scalp_data[:, :mne_raw.n_times], scalp.info)], force_update_info=True)
    mne_raw.reorder_channels(sorted([x for x in mne_raw.ch_names if x not in scalp_chans]) + sorted(scalp_chans))
    rotem_write_edf(mne_raw, f'D:\\Hanna\\D{subj}\\P{subj}_overnightData_fixed.edf')
    print()

# from_mat_to_edf_Hanna_sio(subj='018')
# from_mat_to_edf_Hanna_sio(subj='479')
# from_mat_to_edf_Hanna_sio(subj='489')



def from_mat_to_LFP_Hanna(subj='017'):
    mne_raw = None
    subj_files_list = glob.glob(f'D:\\Hanna\\D{subj}\\Micro\\*')
    sfreq = 1000
    ch_names = pd.read_csv(f'D:\\Hanna\\D{subj}\\montage.csv')
    micro_chans = ch_names[ch_names['IS_MACRO'] == 0].reset_index()
    for i, curr_file in enumerate(subj_files_list):
        try:
            if subj == '018':
                f = sio.loadmat(curr_file)
                data = f['denoised_data']
            else:
                f = h5py.File(curr_file, 'r')
                data = f.get('denoised_data')
            data = np.array(data)
            chan_num = int(curr_file.split('_')[1].replace('.mat', '')) - 1
            curr_name = micro_chans['LOCATION'][chan_num].replace(' ', '')
            info = mne.create_info(ch_names=[curr_name], sfreq=sfreq)
            if mne_raw is None:
                mne_raw = mne.io.RawArray(data[:-10].T, info)
            else:
                if data.size != mne_raw.n_times:
                    data = data[:min(mne_raw.n_times, data.size)]
                mne_raw.add_channels([mne.io.RawArray(data.T, info)], force_update_info=True)
        except Exception as e:
            pass

    mne_raw.reorder_channels(sorted([x for x in mne_raw.ch_names]))
    rotem_write_edf(mne_raw, f'D:\\Hanna\\D{subj}\\P{subj}_overnightData_LFP_fixed.edf')
    print()

# from_mat_to_LFP_Hanna(subj='025')

def from_mat_to_units(subj='025'):
    subj_files_list = glob.glob(f'D:\\Hanna\\D{subj}\\spikesorting\\*')
    ch_names = pd.read_csv(f'D:\\Hanna\\D{subj}\\montage.csv')
    micro_chans = ch_names[ch_names['IS_MACRO'] == 0].reset_index()
    for i, curr_file in enumerate(subj_files_list):
        try:
            f = sio.loadmat(curr_file)
            data = f['cluster_class']
            data = np.array(data).astype('int')
            # index start from 0
            chan_num = int(curr_file.split('CSC')[1].replace('.mat',''))
            curr_name = micro_chans['LOCATION'][chan_num - 1].replace(' ', '')
            for i in range(1, 4):
                data[data[:, 0] == i][:, 1].tolist()
                curr_data = data[data[:, 0] == i]
                if curr_data.size > 0:
                    curr_data = curr_data[:, 1]
                    np.savetxt(f'D:\\Hanna\\D{subj}\\spikesorting\\{curr_name}_unit{i}.csv', curr_data, delimiter=',', fmt="%d")
        except Exception as e:
            pass


# from_mat_to_units()
