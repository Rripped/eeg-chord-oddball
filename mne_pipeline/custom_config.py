study_name = 'chord-oddball'
bids_root = 'D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\ds003570' #<< Modify
deriv_root = "D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\ds003570\\derivatives"
task = 'TASKNAME'   #If this is left empty, it will likely find another task and throw an error
l_freq = 0.5
h_freq = 30.
epochs_tmin = -0.4
epochs_tmax = 1.6
baseline = (0.3, 0.4)
raw_resample_sfreq = 64
ch_types = ['eeg']
data_type = 'meg'
conditions = ['stim']  #<< list of conditions
n_jobs=4  #On biowulf - leave this empty - it will set N_JOBS equal to the cpu cores assigned in sinteractive
on_error = 'continue'  # This is often helpful when doing multiple subjects.  If 1 subject fails processing stops
subjects_dir = 'D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\ds003570'
#crop_runs = [0, 900] #Can be useful for long tasks that are ended early but full file is written.
""" report_evoked_n_time_points = 5
report_stc_n_time_points = 5 """

#D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\data\\ds003570
#D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\ds003570
