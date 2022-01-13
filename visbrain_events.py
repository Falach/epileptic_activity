# not finished!!!!
# not finished!!!!
# not finished!!!!
# not finished!!!!
# not finished!!!!

import mne
import pandas as pd
import numpy as np
from visbrain.gui import Sleep

def peak_index(data, sf, time, hypno):
    index_list = [[row['max_index'] / 4, row['max_index'] / 4] for index, row in spikes_df.iterrows()]
    return np.array(index_list)


edf = 'C:\\Lilach\\396_for_tag_filtered.edf'
raw = mne.io.read_raw_edf(edf)
sf = raw.info['sfreq']
spikes_df = pd.read_csv("C:\\analysis\\396\\best_thresh_detect\\396_LAH1_t5_7_9_for_tag.csv")
stim_amp, stim_grad, stim_env = np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times)), np.zeros((1, raw.n_times))
for evt in spikes_df.values:
    if 'amp' in evt[1]:
        stim_amp[0][evt[4]] = 1
    if 'grad' in evt[1]:
        stim_grad[0][evt[4]] = 1
    if 'env' in evt[1]:
        stim_env[0][evt[4]] = 1

raw.pick_channels(['LAH1'])
info_amp = mne.create_info(['AMP'], raw.info['sfreq'], ['stim'])
info_grad = mne.create_info(['GRAD'], raw.info['sfreq'], ['stim'])
info_env = mne.create_info(['ENV'], raw.info['sfreq'], ['stim'])
amp_raw = mne.io.RawArray(stim_amp, info_amp)
grad_raw = mne.io.RawArray(stim_grad, info_grad)
env_raw = mne.io.RawArray(stim_env, info_env)
raw.load_data()
raw.add_channels([amp_raw, grad_raw, env_raw], force_update_info=True)
# raw.reorder_channels([selected_channel, 'AMP', 'GRAD', 'ENV'])
# if original_sf > 1000:
#     raw.resample(500)
sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names'], downsample=None)
sp.replace_detections('peak', peak_index)
# sp.replace_detections('spindle', spikes_index)
sp.show()
print('finish')


# edf = 'C:\\Lilach\\402_for_tag.edf'
#
# raw = mne.io.read_raw_edf(edf)
#
# data, sf, chan = raw.get_data(), raw.info['sfreq'], raw.info['ch_names']
#
# Sleep(data=data, sf=sf, channels=chan, annotations=raw.annotations).show()
#
# print(1)