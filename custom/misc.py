"""Contains miscellaneous helper functions for the EEG preprocessing pipeline."""

import sys

import ccs_eeg_utils
import os
from contextlib import contextmanager
from dotenv import load_dotenv

from mne_bids import BIDSPath, read_raw_bids

load_dotenv()

TASK = os.getenv("TASK")
bids_root = os.getenv("BIDS_ROOT")


# Context manager to suppress stdout and stderr
@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that suppresses stdout and stderr by piping them to /dev/null.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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


def save_preprocessed_data(file_path, raw):
    """
    Saves the preprocessed EEG data to a file.

    Parameters:
    file_path (str): The path where the preprocessed data will be saved.
    raw (mne.io.Raw): The preprocessed MNE Raw object containing EEG data.
    """
    # Check if file_path ends with .fif extension
    if not file_path.endswith(".fif"):
        file_path += ".fif"

    # Save the data
    try:
        raw.save(file_path, overwrite=True)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def create_bad_json_structure():
    """creates empty json structure for saving bad channels or bad epochs

    Returns:
        dict: dictionary with empty structure for saving bad channels or bad epochs per block per subject
    """
    subjects = {}
    for s in range(1, 41):
        subject_key = f"sub-{s:03d}"
        subjects[subject_key] = {}
        for b in range(1, 9):
            block_key = f"{b}"
            subjects[subject_key][block_key] = []

    return subjects
