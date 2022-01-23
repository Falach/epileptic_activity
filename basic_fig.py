from visbrain.io.rw_hypno import (read_hypno, oversample_hypno)
from visbrain.io import write_fig_hyp
import yasa
import numpy as np
import matplotlib.pyplot as plt
import mne
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize
import pandas as pd

# hypno_file = '/Users/rotemfalach/Documents/University/lab/PAT/ah8/AH8_hypno.txt'
# raw = mne.io.read_raw_edf('/Users/rotemfalach/Documents/University/lab/PAT/ah8/AH8_SLEEP_20201125_155835.edf')
# subject_id = 'ah8'
hypno_file = '/Users/rotemfalach/Documents/University/lab/data/UCLA scoring/P406_hypno.txt'
raw = mne.io.read_raw_edf('/Users/rotemfalach/Documents/University/lab/data/P406_overnightData.edf')
subject_id = 'YZ'
png_file_name = subject_id + '_f3.png'
raw = raw.pick_channels(['EEG C3-REF'])  # pick electrode for spectrogram
separate_NREM_power_spectrum = True  # set false if you want to see N2 and N3 together as NREM


# this is yasa function with some updates
def plot_spectrogram(data, sf, hypno=None, win_sec=30, fmin=0.5, fmax=25, trimperc=2.5, cmap='Spectral_r'):
    __all__ = ['plot_spectrogram']

    # Set default font size to 12
    plt.rcParams.update({'font.size': 12})

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 3600  # Convert t to hours

    # Normalization
    vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    if hypno is None:
        fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
        im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True)
        ax.set_xlim(0, t.max())
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25)
        cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
        return fig
    else:
        hypno = np.asarray(hypno).astype(int)
        assert hypno.ndim == 1, 'Hypno must be 1D.'
        assert hypno.size == data.size, 'Hypno must have the same sf as data.'
        t_hyp = np.arange(hypno.size) / (sf * 3600)
        # Make sure that REM is displayed after Wake
        hypno = pd.Series(hypno).map({-2: -2, -1: -1, 0: 0, 1: 2,
                                      2: 3, 3: 4, 4: 1}).values
        hypno_rem = np.ma.masked_not_equal(hypno, 1)

        fig = plt.figure(constrained_layout = True, figsize=(22, 14))
        grid_spec = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.6])
        ax0 = fig.add_subplot(grid_spec[0, :])
        ax1 = fig.add_subplot(grid_spec[1, :])
        # plt.subplots_adjust(hspace=0.1)
        fig.tight_layout(pad=1)

        # Hypnogram (top axis)
        ax0.step(t_hyp, -1 * hypno, color='k', linewidth=4)
        ax0.step(t_hyp, -1 * hypno_rem, color='r')
        if -2 in hypno and -1 in hypno:
            # Both Unscored and Artefacts are present
            ax0.set_yticks([2, 1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Uns', 'Art', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 2.5)
        elif -2 in hypno and -1 not in hypno:
            # Only Unscored are present
            ax0.set_yticks([2, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Uns', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 2.5)

        elif -2 not in hypno and -1 in hypno:
            # Only Artefacts are present
            ax0.set_yticks([1, 0, -1, -2, -3, -4])
            ax0.set_yticklabels(['Art', 'W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 1.5)
        else:
            # No artefacts or Unscored
            ax0.set_yticks([0, -1, -2, -3, -4])
            ax0.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'])
            ax0.set_ylim(-4.5, 0.5)
        ax0.set_xlim(0, t_hyp.max())
        ax0.set_ylabel('Stage')
        ax0.xaxis.set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)

        # Spectrogram (bottom axis)
        im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True)
        ax1.set_xlim(0, t.max())
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [hrs]')
        return fig, grid_spec, ax0, ax1


# read hypnogram old format (1sec)
hypno, sf_hypno = read_hypno(hypno_file, time=None, datafile=None)

# plot the hypnogram nicely optional to save a file:
# write_fig_hyp(hypno, sf_hypno, grid=True, ascolor=True, file=None)

# this is a built in function to calculate sleep stats:
sleep_stats = yasa.sleep_statistics(hypno, sf_hypno)

# TODO: checkout this epoch fix!!
hypno_file = '/Users/rotemfalach/Documents/University/lab/data/UCLA scoring/P406_hypno.txt'
raw = mne.io.read_raw_edf('/Users/rotemfalach/Documents/University/lab/data/P406_overnightData.edf')
raw.pick_channels(['SEEG RA1-REF', 'SEEG LA1-REF', 'SEEG RAH1-REF', 'SEEG RAH2-REF', 'SEEG LAH1-REF', 'SEEG LAH2-REF', 'EEG C3-REF',
                   'EEG PZ-REF', 'EOG EOG1-REF', 'EOG EOG2-REF'])
raw.resample(1000)

# read hypnogram old format (1sec)
hypno, sf_hypno = read_hypno(hypno_file, time=None, datafile=None)

# plot the hypnogram nicely optional to save a file:
# write_fig_hyp(hypno, sf_hypno, grid=True, ascolor=True, file=None)

# this is a built in function to calculate sleep stats:
sleep_stats = yasa.sleep_statistics(hypno, sf_hypno)

# make raw object into epochs:
epoch_length = 30  # in seconds
# hypno_win = np.asarray((hypno[0:hypno.size:epoch_length]))  # cut the hypnogram to the epochs
event_id = 1  # This is used to identify the events.
dummy_events = mne.make_fixed_length_events(raw, id=event_id, duration=epoch_length)[:len(hypno)]

# incorporate the scoring into the events file:
# dummy_events = dummy_events if len(hypno_win) > len(dummy_events) else dummy_events[:(len(hypno_win) - 1), :]
dummy_events[:, 2] = hypno  # HERE I TRIMMED THE END OF HYPNOGRAM BECAUSE OF THE DISCREPANCY
event_dict = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'art': -1}
# event_dict = {'W': 0, 'N1': 1, 'NREM': 2, 'REM': 4, 'art': -1}
# epoch data into 30sec pieces:
epochs = mne.Epochs(raw, events=dummy_events, event_id=event_dict, tmin=0,
                    tmax=30, baseline=(0, 0), on_missing='ignore')
# epochs.drop(epochs['REM'].selection)
# nrem_raw = mne.io.RawArray(np.concatenate(epochs.get_data(), axis=1), raw.info)
# epochs.save('406_NREM.fif')
# nrem_raw.export('406_NREM.edf')
# write_edf('406_NREM2.edf', nrem_raw)
# rotem_edf(nrem_raw, '406_NREM3.edf')


# plot spectrogram with hypnogram:
data_out = raw.get_data()
hypno_up = oversample_hypno(hypno, data_out.shape[1])
# spectrogram_fig, gs, ax0, ax1 = plot_spectrogram(data_out[0, :], raw.info['sfreq'], hypno_up, trimperc=10)
# ax0.set(title=subject_id)

stage_spect_trls = [None] * len(event_dict)
stage_spect_avg = [None] * len(event_dict)
stage_spect_std = [None] * len(event_dict)
if separate_NREM_power_spectrum:
    stages_dict = {'W': 0, 'N2': 2, 'N3': 3, 'REM': 4}
else:
    stages_dict = {'W': 0, 'N2': 2, 'REM': 4}

for ind, stage in enumerate(stages_dict):
    if epochs[stage].__len__() > 0:  # if not empty
        if separate_NREM_power_spectrum or stage != 'N2':
            psds, freqs = mne.time_frequency.psd_multitaper(epochs[stage], fmin=0.5, fmax=40, n_jobs=1)
        else:
            psds2, freqs2 = mne.time_frequency.psd_multitaper(epochs[stage], fmin=0.5, fmax=40, n_jobs=1)
            psds3, freqs3 = mne.time_frequency.psd_multitaper(epochs['N3'], fmin=0.5, fmax=40, n_jobs=1)
            psds = np.concatenate((psds2, psds3), axis=0)

        stage_spect_trls[ind] = psds

        psds_mean = psds.mean(0).mean(0)
        psd_db = 10. * np.log10(psds_mean)

        stage_spect_avg[ind] = psd_db

        psds_std = psds.mean(0).std(0)
        stage_spect_std[ind] = psds_std


# plotting spectral figures:
stage_color = 'rygbk'
# stages_dict = {key: value for key, value in event_dict.items() if value >= 0}
# stages_dict = {'W': 0, 'N2': 2, 'REM': 4}
ax2 = spectrogram_fig.add_subplot(gs[2, 0], aspect=0.8)

for i, stage in enumerate(stages_dict):  # runs on stages
    if stage_spect_avg[i] is not None:
        ax2.plot(freqs, stage_spect_avg[i], color=stage_color[i], label=stage)
        # plt.fill_between(freqs, stage_spect_avg[i] - stage_spect_std[i], stage_spect_avg[i] + stage_spect_std[i],
        #                  color=stage_color[i], alpha=.2)


ax2.set_title('PSD according to stages', fontsize='12')
ax2.set_ylabel('Power Spectral Density (dB)', fontsize='10')
ax2.set_xlabel('Frequency (Hz)', fontsize='10')
ax2.legend(['wake', 'N2', 'N3', 'REM'] if separate_NREM_power_spectrum else ['wake', 'NREM', 'REM'],
           prop={'size': 10}, loc='upper right')

# statistics table
for key, value in sleep_stats.items():
    # take only two numbers after the dot
    sleep_stats[key] = "%.2f" % value

ax3 = spectrogram_fig.add_subplot(gs[2, 1])
ax3.axis("off")
ax3.axis('tight')
table = ax3.table(cellText=[[sleep_stats['TIB']], [sleep_stats['TST']], [sleep_stats['SE']], [sleep_stats['SOL']],
                            [sleep_stats['WASO']], [sleep_stats['%N1']], [sleep_stats['%N2']], [sleep_stats['%N3']],
                            [sleep_stats['%NREM']], [sleep_stats['%REM']]],
                  rowLabels=['Total time in bad, min', 'Total sleep, min', 'Sleep efficiency, %', 'Sleep latency, min',
                             'WASO', 'Stage 1, %', 'Stage 2, %', 'SWS, %', 'NREM, %', 'REM, %'],
                  colWidths=(0.3, 1), loc='center')
table.set_fontsize(16)
table.scale(1, 2)

plt.savefig(png_file_name, bbox_inches='tight')
print('finish')
