"""
This script will convert mff to edf file according to config, example:
python mff_to_edf.py "example_file.mff"

This command will generate example_file.edf at the same path as the input file

author: Rotem Falach
"""
import pyedflib
from pyedflib._extensions._pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta
import mne
from config import *
from mne import viz
import os
import numpy as np
from matplotlib import pyplot as plt


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


# TODO: use mnelab code? https://github.com/cbrnr/mnelab/blob/main/mnelab/io/writers.py
def write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=True):
    conversion_time = datetime.now()
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')

    # static settings
    has_annotations = True if len(mne_raw.annotations) > 0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607

    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])

    if tmin:
        date += timedelta(seconds=tmin)
    first_sample = int(sfreq * tmin)
    last_sample = int(sfreq * tmax) if tmax is not None else None

    # convert data
    channels = mne_raw.get_data(picks,
                                start=first_sample,
                                stop=last_sample)

    # convert to microvolts to scale up precision
    # channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []

        ch_idx = range(n_channels) if picks is None else picks
        for i in ch_idx:
            ch_dict = {'label': mne_raw.ch_names[i],
                       'dimension': 'uV',
                       'sample_rate': sfreq,
                       'physical_min': channels[i].min(),
                       'physical_max': channels[i].max() + 1,
                       'digital_min': dmin,
                       'digital_max': dmax,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)

        # TODO: save configuration in header (electrodes, right_set, etc)
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        f.close()
        print('conversion time:')
        print(datetime.now() - conversion_time)
    return True


def organize_data(file_name):
    main_channels_time = datetime.now()
    egi_full_montage = mne.channels.make_standard_montage('GSN-HydroCel-257')

    # load the data, apply sensor locations
    raw = mne.io.read_raw_egi(file_name, verbose=True)
    raw.rename_channels({'E257': 'Cz'})  # replace raw Cz naming to fit montage
    raw.set_montage(egi_full_montage)

    # pick the relevant channels
    eeg_montage = raw.copy().pick_channels(selected_set[0])
    emg_montage = raw.copy().pick_channels(emg_set[0])

    # resample to smaller sample rate
    eeg_montage.resample(target_sample_rate)
    emg_montage.resample(target_sample_rate)

    # filter channels
    eeg_montage.filter(eeg_bp_freq[0], eeg_bp_freq[1])
    emg_montage.filter(emg_bp_freq[0], emg_bp_freq[1])

    # order channels according to selected set, MNE default is alphabetical
    eeg_montage.reorder_channels(selected_set[0])

    # rename channels
    eeg_montage.rename_channels({id: name for (id, name) in zip(selected_set[0], selected_set[1])})
    emg_montage.rename_channels({id: name for (id, name) in zip(emg_set[0], emg_set[1])})

    # set EOG channels as MNE type "ecog" for re-refrencing
    eeg_montage.set_channel_types({selected_set[1][i]: 'ecog' for i in range(2)})

    # set EEG channels as MNE type 'eeg' for re-refrencing
    eeg_montage.set_channel_types({selected_set[1][i]: 'eeg' for i in range(3, 6)})

    # set EMG channels as MNE type "seeg" for re-refrencing
    emg_montage.set_channel_types({emg_set[1][i]: 'seeg' for i in range(2)})

    # re-reference channels
    eeg_montage.set_eeg_reference(ref_channels=[selected_set[1][2]], ch_type='ecog')
    eeg_montage.set_eeg_reference(ref_channels=[selected_set[1][6]], ch_type='eeg')
    emg_montage.set_eeg_reference(ref_channels=[emg_set[1][1]], ch_type='seeg')

    # Return to the accurate type
    eeg_montage.set_channel_types({selected_set[1][i]: 'eog' for i in range(2)})
    emg_montage.set_channel_types({channel: 'emg' for channel in emg_set[1]})

    # remove reference electrodes
    eeg_montage.drop_channels([selected_set[1][2], selected_set[1][6]])
    emg_montage.drop_channels([emg_set[1][1]])

    # Append all channels together
    all_channels = eeg_montage.copy().add_channels([emg_montage], force_update_info=True)
    print("finish 6 main channels, time:")
    print(datetime.now() - main_channels_time)

    # PIB channels
    if contain_respiration:
        resp_time = datetime.now()
        resp_montage = raw.copy().pick_channels(resp_set[0])
        resp_montage.resample(target_sample_rate)
        resp_montage.filter(airflow_bp_freq[0], airflow_bp_freq[1], picks=[resp_set[0][0]])\
            .filter(snore_bp_freq[0], snore_bp_freq[1], picks=[resp_set[0][1]])\
            .filter(belt_bp_freq[0], belt_bp_freq[1], picks=resp_set[0][2:])
        resp_montage.rename_channels({id: name for (id, name) in zip(resp_set[0], resp_set[1])})

        # set limits to airflow channel, clipping and scaling values
        resp_montage.apply_function(np.clip, resp_set[1], None, 1, True, None, -1e-4, 1e-4)
        resp_montage.apply_function(lambda x: x * 20, resp_set[1], None, 1, True, None)
        resp_montage.apply_function(lambda x: x * 5, resp_set[1][1], None, 1, True, None)

        # finish respiration montage and append it to the eeg montage
        all_channels.add_channels([resp_montage], force_update_info=True)
        print("finish 4 respiration channels, time:")
        print(datetime.now() - resp_time)

    if contain_spo2:
        spo2_time = datetime.now()
        spo2_montage = raw.copy().pick_channels(spo2_set[0])
        spo2_montage.resample(target_sample_rate)
        spo2_montage.filter(spo2_bp_freq[0], spo2_bp_freq[1], picks=[spo2_set[0][0]])\
            .filter(pleth_bp_freq[0], pleth_bp_freq[1], picks=[spo2_set[0][1]])
        spo2_montage.rename_channels({id: name for (id, name) in zip(spo2_set[0], spo2_set[1])})

        # old scaling
        # min_value = 0.03
        # max_value = 2
        # scale_func = lambda x: ((abs(x - min_value) / abs(max_value - min_value))) / 100
        # spo2_montage.apply_function(scale_func, spo2_set[1][0], None, 1, True, None)
        # spo2_montage.apply_function(lambda x: x / 10, spo2_set[1][1], None, 1, True, None)

        # new solution
        min_value = -7.12e-05
        max_value = spo2_montage.get_data()[0].max() * 0.92
        median = np.median(spo2_montage.get_data()[0])
        # for some reason sometimes the graph is upside down
        if abs(median - min_value) < abs(median - max_value):
            spo2_montage.apply_function(lambda x: x * (-1), spo2_set[1][0], None, 1, True, None)
            min_value *= -1
            max_value *= -1
            scale_func = lambda x: (1 - (abs(x - min_value) / abs(max_value - min_value))) / 10000
        else:
            scale_func = lambda x: (abs(x - min_value) / (abs(max_value - min_value))) / 10000
        spo2_montage.apply_function(scale_func, spo2_set[1][0], None, 1, True, None)
        spo2_montage.apply_function(lambda x: x / 10, spo2_set[1][1], None, 1, True, None)
        # if not for_alise:
        #     spo2_montage.apply_function(lambda x: x / 10, spo2_set[1][1], None, 1, True, None)

        # # export csv file with spo2 values
        # np.savetxt(file_name.split('.mff')[0] + '_spo2.csv', spo2_montage.get_data()[0] * 1000000, delimiter=",", fmt='%f')

        # viz.plot_raw(spo2_montage)
        # plt.plot(spo2_montage.get_data()[0])
        all_channels.add_channels([spo2_montage], force_update_info=True)
        print("finish 2 spo2 channels, time:")
        print(datetime.now() - spo2_time)

    # viz.plot_raw(all_channels, duration=1000, scalings=dict(bio=10e-5, eeg=10e-5, eog=10e-5, emg=10e-5))

    # to check a new feature on a small amount of data
    # selection = all_channels.copy().crop(tmin=100, tmax=200)  # get 100 seconds

    return all_channels


def main():
    # Save start time just for checking speed
    start_time = datetime.now()

    # Get file name as a param from cmd
    # mff_file_name = sys.argv[1]
    side = '_right' if selected_set == right_set else '_left'
    edf_file_name = mff_file_path.split('.mff')[0] + side + '.edf'

    # Get the relevant electrodes and apply some filters
    mff_data = organize_data(mff_file_path)

    # Convert
    write_edf(mne_raw=mff_data, fname=edf_file_name, overwrite=True)

    # TODO: save fif (set flag in config)
    # Check if the edf file is readable
    # raw2 = mne.io.read_raw_edf(edf_file_name, stim_channel='auto')

    # Printing conversion duration
    print(datetime.now())
    print("total time")
    print(datetime.now() - start_time)


if __name__ == "__main__":
    main()