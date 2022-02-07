import mne
from visbrain.gui import Sleep
import pandas as pd
import numpy as np

# def spikes_index(data, sf, time, hypno):
#     index_list = [[row['max_index'] / 4, row['max_index'] / 4] for index, row in spikes_df.iterrows()]
#     return np.array(index_list)

# edf = 'C:\\UCLA\\P406_staging_PSG_and_intracranial_Mref_correct.txt.edf'
# edf = 'C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\402\\402_for_tag.edf'
# edf = "C:ֿֿֿֿ\\analysis\\396\\396_LAH1_100_samples.fif"
subj = '406'
edf = f'C:\\UCLA\\P{subj}_overnightData.edf'
depth_epochs = pd.read_csv(f'C:\\repos\\spikes_notebooks\\spike_index_{subj}.csv')
scalp_epochs = pd.read_csv(f'C:\\repos\\spikes_notebooks\\scalp_index_{subj}.csv')
raw = mne.io.read_raw_edf(edf).pick_channels(['EOG1', 'EOG2', 'C3', 'C4', 'PZ', 'RAH1', 'RAH2', 'LAH1', 'LAH2']).resample(1000)
raw = mne.set_bipolar_reference(raw, 'RAH1', 'RAH2', ch_name='RAH1-RAH2', drop_refs=False)
raw = mne.set_bipolar_reference(raw, 'LAH1', 'LAH2', ch_name='LAH1-LAH2', drop_refs=False)

depth = np.zeros((1, raw.n_times))
for evt in depth_epochs.values:
        depth[0][250 * evt[0]: 250 * evt[0] + 250] = 1

scalp = np.zeros((1, raw.n_times))
for evt in scalp_epochs.values:
        scalp[0][250 * evt[0]: 250 * evt[0] + 250] = 1

# raw.pick_channels(['LAH1'])
info_depth = mne.create_info(['depth'], raw.info['sfreq'], ['stim'])
info_scalp = mne.create_info(['scalp'], raw.info['sfreq'], ['stim'])
depth_raw = mne.io.RawArray(depth, info_depth)
scalp_raw = mne.io.RawArray(scalp, info_scalp)
raw.load_data()
raw.add_channels([depth_raw, scalp_raw], force_update_info=True)
raw.reorder_channels(['EOG1', 'EOG2', 'C3', 'C4', 'PZ', 'RAH1', 'RAH2', 'RAH1-RAH2', 'LAH1', 'LAH2', 'LAH1-LAH2', 'depth', 'scalp'])

sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names'], downsample=1000)
# sp.replace_detections('spindle', spikes_index)
sp.show()
print('finish')

# 396, 398, 402, 406, 416 - good depth detection
# scalp:
# 398- can't really see spikes on EOG, c3 pretty bad and somtime also c4 pz
# 406- C3 pretty suck, somtime others
# 415- wierd stuff, shitty scalp but ok depth detection, somtime all channels look the same (scalp+depth), why?
# 416- very low amp scalp channels while spikes and the opposite, why? maybe cortex activity?
