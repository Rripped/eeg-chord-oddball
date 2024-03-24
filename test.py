import mne
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "../")
import ccs_eeg_utils
import numpy as np
import pandas as pd
import mne.preprocessing as prep
import os
import sklearn 
from contextlib import contextmanager
from autoreject import AutoReject, get_rejection_threshold
import json

from mne_bids import (BIDSPath, read_raw_bids, write_raw_bids, inspect_dataset)
import auc

from mne_icalabel import label_components
from pyprep.find_noisy_channels import NoisyChannels

# path where dataset is stored
bids_root = "./data/ds003570/"
TASK = 'AuditoryOddballChords'
SUBJECT = '038'
SUPRESS_BIDS_OUTPUT = True
PROMPT_BADS = False
USE_ICA_JSON = False
ICA_MANUAL = False
Z_SCORE_REJECT = False
PYPREP_REJECT = True
AUTOREJECT = False


def read_raw_data(subject_id):
    """
    Reads the raw EEG data from the BIDS dataset.

    Parameters:
    subject_id (str): The subject ID.

    Returns:
    raw (mne.io.Raw): The MNE Raw object containing EEG data.
    """
    bids_path = BIDSPath(
        subject=subject_id, datatype="eeg", suffix="eeg", task=TASK, root=bids_root
    )

    raw = read_raw_bids(bids_path)

    ccs_eeg_utils.read_annotations_core(bids_path, raw)
    raw.load_data()

    return raw, bids_path

#raw = mne.io.read_raw_fif("D:\\Uni\\1_Semester\\EEG\\eeg-chord-oddball\\data\\processed_002_raw.fif", preload=True)

raw, _ = read_raw_data("038")

print("Type: ")
print(type(raw))