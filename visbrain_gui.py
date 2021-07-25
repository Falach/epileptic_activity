import mne
from visbrain.gui import Sleep

edf = 'C:\\UCLA\\P406_staging_PSG_and_intracranial_Mref_correct.txt.edf'
raw = mne.io.read_raw_edf(edf)

data, sf, chan = raw.get_data(), raw.info['sfreq'], raw.info['ch_names']

Sleep(data=data, sf=sf, channels=chan).show()

print(1)
