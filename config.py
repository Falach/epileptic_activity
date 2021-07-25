target_sample_rate = 250

# TODO: add notch filter before (50Hz) - all channels (also for respiration)
# filters
eeg_bp_freq = [0.3, 40]
emg_bp_freq = [10, 100]
airflow_bp_freq = [0, 15]  # channel 258 (to check) or airflow
snore_bp_freq = [10, 100]
belt_bp_freq = [0.1, 15]
pleth_bp_freq = [0.1, 30]
spo2_bp_freq = [0, 30]

# this is the right set that goes with LM
right_set = [['E252', 'E10', 'E190', 'E224', 'E183', 'E150', 'E105'],
             ['EOG E1-RM', 'EOG E2-RM', 'RM', 'EEG F4-LM', 'EEG C4-LM', 'EEG O2-LM', 'LM']]

# this is the  left set referenced to RM, while EOG referenced to LM
left_set = [['E226', 'E46', 'E104', 'E35', 'E59', 'E116', 'E201'],
            ['EOG E1-LM', 'EOG E2-LM', 'LM', 'EEG F3-RM', 'EEG C3-RM', 'EEG O1-RM', 'RM']]

emg_set = [['E243', 'E240'],
           ['EMG', 'NOTHING']]

resp_set = [['Airflow', 'Snore', 'Chest', 'Abdomen'],
            ['Flow Airflow', 'Flow Snore', 'Effort Chest-Be', 'Effort Abdomen-']]

spo2_set = [['SpO2', 'Pleth'],
            ['SaO2 SPO2', 'Pulse Pleth']]

# mic = [['microphone']]

# change this variable to your mff file path
# mff_file_path = '/Users/rotemfalach/Documents/University/lab/VZ9_sleep_20190718_124025.mff'
mff_file_path = 'C:\\Elad 072021\\YG-OB9067_20210412_223729_2.mff'

# choose your set
selected_set = left_set

# choose scoring method (different scaling between vizbarin/edf browser and alise)
for_alise = True

# turn on/off PIB channels
contain_respiration = False
contain_spo2 = True