import unittest
import os
import numpy as np
import mne
from custom.misc import *
from custom.preprocessing import *

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Load raw data for testing
        self.raw, self.bids_path = read_raw_data(SUBJECT)
        self.fif = mne.io.read_raw_fif("D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\data\\processed_002_raw.fif", preload=True)

    def test_read_raw_data(self):
        # Test if raw data is loaded successfully
        self.assertIsInstance(self.raw, mne.io.eeglab.eeglab.RawEEGLAB)

    def test_basic_preprocessing(self):
        # Create a Raw object with a sampling frequency > 128 Hz
        sfreq = 256  # Sampling frequency
        n_channels = 10  # Number of channels
        duration = 10  # Duration in seconds
        raw = mne.io.RawArray(data=[[0]*sfreq]*n_channels, info=mne.create_info(ch_names=['ch'+str(i) for i in range(n_channels)], sfreq=sfreq))
        raw.set_channel_types({'ch0': 'ecg', 'ch1': 'eog'})
        
        # Apply basic preprocessing
        preprocessed_raw = basic_preprocessing(raw)
        
        # Assert that the sampling frequency is now 128 Hz
        self.assertEqual(preprocessed_raw.info['sfreq'], 256)
        self.assertTrue(all(ch_type in ['eeg', 'stim'] for ch_type in preprocessed_raw.get_channel_types().values()))
        self.assertEqual(raw._data.shape, preprocessed_raw._data.shape)
        self.assertEqual(preprocessed_raw.info['custom_ref_applied'], 'average')

    def test_preprocessing(self):
        # Test preprocessing steps
        preprocessed_raw = preprocessing(self.raw)
        self.assertEqual(preprocessed_raw.info['sfreq'], 128)
        self.assertEqual(len(preprocessed_raw.info['ch_names']), 64)

    def test_save_preprocessed_data(self):
        # Test saving preprocessed data
        save_path = "./data/fifs/test_preprocessed_raw.fif"
        save_preprocessed_data(save_path, self.raw)
        self.assertTrue(os.path.exists(save_path))

    def test_get_ica(self):
        # Test ICA computation
        ica_raw = get_ica(self.fif, {}, 1, mne.channels.make_dig_montage())
        self.assertIsInstance(ica_raw, mne.io.Raw)

    def test_split_in_blocks(self):
        # Test splitting raw data into blocks
        blocks = split_in_blocks(self.raw)
        self.assertIsInstance(blocks, list)
        self.assertTrue(all(isinstance(block, mne.io.eeglab.eeglab.RawEEGLAB) for block in blocks))

    def test_mark_bad_channels_by_z_score(self):
        # Test marking bad channels based on z-score
        bad_channels = mark_bad_channels_by_z_score(self.raw)
        self.assertIsInstance(bad_channels, list)
        self.assertTrue(all(isinstance(ch, str) for ch in bad_channels))

    def test_get_epochs_from_events(self):
        # Test epoching based on events
        standard_epochs = get_epochs_from_events(self.fif, '_S')
        self.assertIsInstance(standard_epochs, mne.epochs.Epochs)

    def test_apply_autoreject_info(self):
        # Test applying autoreject information to epochs
        epochs = get_epochs_from_events(self.fif, '_S')
        autoreject_info = {'threshes': {}, 'reject_log': {}}
        epochs_clean = apply_autoreject_info(epochs, autoreject_info)
        self.assertIsInstance(epochs_clean, mne.Epochs)

    def test_epoch_rejection(self):
        # Test epoch rejection based on criteria
        epochs = get_epochs_from_events(self.fif, '_S')
        shape = (len(epochs),) + epochs.get_data()[0].shape
        epochs_rejected = epoch_rejection(epochs.get_data(), shape)
        self.assertIsInstance(epochs_rejected, np.ndarray)

    def test_create_bad_json_structure(self):
        # Test creating JSON structure for bad channels
        bad_structure = create_bad_json_structure()
        self.assertIsInstance(bad_structure, dict)
        self.assertTrue(all(isinstance(sub, dict) for sub in bad_structure.values()))

    def test_set_bad_channels_from_json(self):
        # Test setting bad channels from JSON structure
        blocks = split_in_blocks(self.raw)
        bad_json = create_bad_json_structure()
        blocks_with_bad_channels = set_bad_channels_from_json(blocks, bad_json)
        self.assertTrue(all(isinstance(block.info['bads'], list) for block in blocks_with_bad_channels))

    def tearDown(self):
        # Clean up any temporary files created during testing
        save_path = "./data/fifs/test_preprocessed_raw.fif"
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main()

# basic_preprocessing, 
