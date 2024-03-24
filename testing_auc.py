import numpy as np
import unittest
from auc import (
    generate_AUC_ROC,
    generate_AUC_ROC_normalized,
    generate_AUC_ROC_sliding_window,
    generate_AUC_ROC_sliding_window_normalized,
    generate_AUC_ROC_legacy,
    generate_AUC_ROC_legacy_sw,
    generate_forward_model_sw,
    moving_window_smoothing
)

class TestFunctions(unittest.TestCase):

    def setUp(self):
        # Set up test arrays
        self.epoch_standard = np.random.rand(2, 3, 10)
        self.epoch_deviant = np.random.rand(2, 3, 10)
        self.list_of_interest_points = [0, 1, 2]
        self.window_length = 3
        self.stepsize = 1

    def test_generate_AUC_ROC(self):
        auc_roc = generate_AUC_ROC(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc, list)
        self.assertEqual(len(auc_roc), 8)  # Adjust length based on inputs

    def test_generate_AUC_ROC_normalized(self):
        auc_roc_normalized = generate_AUC_ROC_normalized(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc_normalized, list)
        self.assertEqual(len(auc_roc_normalized), 8)  # Adjust length based on inputs

    def test_generate_AUC_ROC_sliding_window(self):
        auc_roc_sliding_window = generate_AUC_ROC_sliding_window(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc_sliding_window, list)
        self.assertEqual(len(auc_roc_sliding_window), 10)  # Adjust length based on inputs

    def test_generate_AUC_ROC_sliding_window_normalized(self):
        auc_roc_sliding_window_normalized = generate_AUC_ROC_sliding_window_normalized(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc_sliding_window_normalized, list)
        self.assertEqual(len(auc_roc_sliding_window_normalized), 10)  # Adjust length based on inputs

    def test_generate_AUC_ROC_legacy(self):
        auc_roc_legacy = generate_AUC_ROC_legacy(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc_legacy, list)
        self.assertEqual(len(auc_roc_legacy), 8)  # Adjust length based on inputs

    def test_generate_AUC_ROC_legacy_sw(self):
        epoch_standard_data = moving_window_smoothing(self.epoch_standard, self.window_length)
        length = len(epoch_standard_data[0, 0])
        length= length - self.window_length + 1
        auc_roc_legacy_sw = generate_AUC_ROC_legacy_sw(
            self.epoch_standard, self.epoch_deviant, self.window_length, self.stepsize
        )
        self.assertIsInstance(auc_roc_legacy_sw, list)
        self.assertEqual(len(auc_roc_legacy_sw), length)  # Adjust length based on inputs

    def test_generate_forward_model_sw(self):
        forward_model_sw = generate_forward_model_sw(
            self.epoch_standard, self.epoch_deviant, self.list_of_interest_points, self.window_length, self.stepsize
        )
        self.assertIsInstance(forward_model_sw, np.ndarray)

if __name__ == '__main__':
    unittest.main()
