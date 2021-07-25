import scipy.io as sio
import mat73
import mne
# from visbrain.gui import Sleep
import pandas as pd
import numpy as np
from mnelab.io.writers import write_edf
from mff_to_edf import write_edf as rotem_write_edf


def for_firas_396(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P396_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    # notice REF1(!) and -6 index
    channels_list_original = ['SEEG REC1-REF1', 'SEEG REC2-REF1', 'SEEG RAH1-REF1', 'SEEG RAH2-REF1',
                     'SEEG LA1-REF1', 'SEEG LA2-REF1', 'SEEG LEC1-REF1', 'SEEG LEC2-REF1', 'SEEG LAH1-REF1', 'SEEG LAH2-REF1',
                              'SEEG LMH1-REF1', 'SEEG LMH2-REF1']
    channels_list_bipolar_original = [x[:-6] + str(int(x[-6]) + 1) + '-REF1' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF1', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF1', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF1', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('396_bipolar.edf' if file_name is None else file_name, raw_bi)
    # write_edf('396_fix.edf', raw.crop(tmax=200))


# for_firas_396()


def for_firas_398(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P398_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG R'
                              'A1-REF', 'SEEG RA2-REF', 'SEEG REC1-REF', 'SEEG REC2-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF',
                     'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('398_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_398()


def for_firas_402(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P402_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG RA1-REF', 'SEEG RA2-REF', 'SEEG REC1-REF', 'SEEG REC2-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF',
                     'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LEC1-REF', 'SEEG LEC2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('402_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_402()


def for_firas_405(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P405_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG RA1-REF', 'SEEG RA2-REF', 'SEEG RAH2-REF',
                            'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('405_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_405()


def for_firas_406(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P406_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG RA1-REF', 'SEEG RA2-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF',
                              'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('406_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_406()

def for_firas_415(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P415_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG RA1-REF', 'SEEG RA2-REF', 'SEEG REC1-REF', 'SEEG REC2-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF',
                     'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LEC1-REF', 'SEEG LEC2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]

    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('415_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_415()


def for_firas_416(crop_start=None, file_name=None):
    edf = 'C:\\UCLA\\P416_overnightData.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['SEEG RA1-REF', 'SEEG RA2-REF', 'SEEG REC1-REF', 'SEEG REC2-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF',
                     'SEEG LA1-REF', 'SEEG LA2-REF', 'SEEG LEC1-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF']
    channels_list_bipolar_original = [x[:-5] + str(int(x[-5]) + 1) + '-REF' for x in channels_list_original]
    # LEC2 is dirty
    channels_list_bipolar_original[8] = 'SEEG LEC3-REF'
    channels_list = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace('SEEG ', '').replace('-REF', '') for x in channels_list_bipolar_original]

    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace('SEEG ', '').replace('-REF', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()

    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    # raw_bi.plot()

    write_edf('416_bipolar.edf' if file_name is None else file_name, raw_bi)


# for_firas_416()

def for_firas_34(crop_start=None, file_name=None):
    edf = 'C:\\Matlab\\Patient93_LTM-1_t3.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['RA 01', 'RA 02', 'RAH 01', 'RAH 02', 'LAH 01', 'LAH 02']
    channels_list_bipolar_original = [x[:-1] + str(int(x[-1]) + 1) for x in channels_list_original]
    channels_list = [x.replace(' 0', '') for x in channels_list_original]
    # RAH3 is dirty
    channels_list_bipolar_original[3] = 'RAH 04'
    channels_list_bipolar = [x.replace(' 0', '') for x in channels_list_bipolar_original]
    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace(' 0', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()
    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    rotem_write_edf(raw_bi, '416_bipolar.edf' if file_name is None else file_name)


def for_firas_36(crop_start=None, file_name=None):
    edf = 'C:\\Matlab\\D036_15_01_21a.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['RA 02', 'RMH 01', 'RMH 02', 'LA 02', 'LH 01', 'LH 02']
    channels_list_bipolar_original = [x[:-1] + str(int(x[-1]) + 1) for x in channels_list_original]
    channels_list = [x.replace(' 0', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace(' 0', '') for x in channels_list_bipolar_original]
    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace(' 0', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()
    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    rotem_write_edf(raw_bi, '416_bipolar.edf' if file_name is None else file_name)


def for_firas_37(crop_start=None, file_name=None):
    edf = 'C:\\Matlab\\D037_28_01_21a.edf'
    raw = mne.io.read_raw_edf(edf)
    channels_list_original = ['RA 01', 'RA 02', 'RH 01', 'RH 02', 'LAH 01', 'LAH 02']
    channels_list_bipolar_original = [x[:-1] + str(int(x[-1]) + 1) for x in channels_list_original]
    channels_list = [x.replace(' 0', '') for x in channels_list_original]
    channels_list_bipolar = [x.replace(' 0', '') for x in channels_list_bipolar_original]
    raw.pick_channels(list(set(channels_list_original) | set(channels_list_bipolar_original)))
    channels_mapping = {x: x.replace(' 0', '') for x in raw.ch_names}
    raw.rename_channels(channels_mapping)
    if crop_start is not None:
        raw.crop(tmin=crop_start * 60, tmax=crop_start * 60 + 16 * 60)
        raw.resample(250)
    raw.load_data()
    raw_bi = mne.set_bipolar_reference(raw, anode=channels_list, cathode=channels_list_bipolar, drop_refs=False)
    rotem_write_edf(raw_bi, '416_bipolar.edf' if file_name is None else file_name)


