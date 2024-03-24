"""Contains relevant functions for preprocessing the EEG data used in the project."""

import os

import mne
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from mne_icalabel import label_components

load_dotenv()

USE_ICA_JSON = os.getenv("USE_ICA_JSON", "False") == "True"
ICA_MANUAL = os.getenv("ICA_MANUAL", "False") == "True"
SUBJECT = os.getenv("SUBJECT")


def basic_preprocessing(raw):
    """
    base preprocessing of the given raw data.

    Args:
        raw (mne.io.Raw): The raw data to preprocess

    Returns:
        mne.io.Raw: The preprocessed raw data
    """
    # downsampling to 128 Hz (paper even did 64 Hz)
    if raw.info["sfreq"] > 128:
        raw.resample(128)

    # Set channel types to EEG if not already set
    if not all(ch_type in ["eeg", "stim"] for ch_type in raw.get_channel_types()):
        eeg_channel_names = raw.ch_names
        channel_types = {name: "eeg" for name in eeg_channel_names}
        raw.set_channel_types(channel_types)

    # band-pass filter between 0.5 Hz and 30 Hz
    raw.filter(0.5, 30, fir_design="firwin")

    # rereferencing to the average activity of all electrodes
    raw.set_eeg_reference("average", projection=True)

    return raw  # preprocessed raw data


def get_ica(
    data,
    ica_bads,
    block_idx,
    montage,
    subject=SUBJECT,
    use_ica_json=USE_ICA_JSON,
    manual_selection=ICA_MANUAL,
):
    """
    Perform ICA on the given data. The 3 possible options are:
        1) Use the Data in the corresponding JSON File (USE_ICA_JSON=True)
        2) Label manual the components to exclude (USE_ICA_JSON=False and ICA_MANUAL=True)
        3) Automated approach using mne_icalabel (USE_ICA_JSON=False and ICA_MANUAL=False)
    Save all rejected ICA channels to ica_bads.

    Args:
        data (mne.raw): The data on which the ICA should be computed
        ica_bads (dict): datastructure to create the JSON File from
        block_idx (int): Index of the current block
        montage (mne.channels.DigMontage): Postions of the Electrodes

    Returns:
        mne.raw: Data on which ICA is performed
    """
    data.set_montage(montage)
    # set random_state to a constant such that the ICA stays the same for multiple runs
    ica = mne.preprocessing.ICA(method="fastica", random_state=0)

    ica.fit(data, verbose=True)

    if use_ica_json:
        exclude_components = ica_bads[f"sub-{subject}"][block_idx]
    else:

        # gen automated labeling
        ic_labels = label_components(data, ica, method="iclabel")

        if manual_selection:

            # plot components and corresponding estimated label
            ica_n_components = ica.n_components_
            n_components = 64
            fig, axes = plt.subplots(nrows=7, ncols=10, figsize=(15, 10))
            axes = axes.flatten()

            # plot per component
            for i, ax in enumerate(axes[:ica_n_components]):
                if i < n_components:
                    ica.plot_components(picks=i, show=False, axes=ax)
                    ax.text(
                        0.5,
                        1,
                        ic_labels["labels"][i],
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    ax.set_visible(False)
            plt.show(block=True)

            # user interaction for manual labeling
            input_str = input(
                "Enter index of the components to be separated by space: "
            )

            # Converting input string to a list of integers
            exclude_components = input_str.split()
            exclude_components = [int(num) for num in exclude_components]
        else:

            # Exclude all components which are not labeled as brain or other
            exclude_components = [
                idx
                for idx, label in enumerate(ic_labels["labels"])
                if label not in ["brain", "other"]
            ]

        # save the bad components to list
        ica_bads[f"sub-{SUBJECT}"][block_idx] = exclude_components

    print("List of components:", exclude_components)
    ica.exclude = exclude_components
    reconst_raw = data.copy()

    # apply with excluded components
    reconst_raw = ica.apply(reconst_raw)
    return reconst_raw


def split_in_blocks(raw):
    """split the raw data into blocks based on the boundary events (should be 8 blocks for the given data)

    Args:
        raw (mne.io.Raw): The raw data to split

    Returns:
        list: List of the blocks (each a mne.io.Raw object)
    """
    events, event_id = mne.events_from_annotations(raw)

    # identify indices of boundary events
    boundary_events = events[events[:, 2] == event_id["STATUS:boundary"], 0]

    blocks = []
    start_idx = 0
    for end_idx in boundary_events:
        block = raw.copy().crop(tmin=raw.times[start_idx], tmax=raw.times[end_idx])
        blocks.append(block)
        start_idx = end_idx

    block = raw.copy().crop(tmin=raw.times[start_idx])
    blocks.append(block)

    return blocks


def mark_bad_channels_by_z_score(raw_data, threshold=8.0, window_size=10000):
    """
    Identify bad channels in raw data based on amplitude.
    Channels with z-score > threshold are marked as bad.

    Parameters:
    raw_data (mne.io.Raw): The raw data.
    threshold (float): The z-score threshold to use.

    Returns:
    list: List of bad channels.
    """
    data, times = raw_data[:, :]
    bad_channels = set()

    # iterate over the data in windows
    for start in range(0, data.shape[1], window_size):
        end = min(start + window_size, data.shape[1])
        window_data = data[:, start:end]

        channel_means = np.mean(window_data, axis=1)
        channel_stds = np.std(window_data, axis=1)
        z_scores = np.abs(
            (window_data - channel_means[:, None]) / channel_stds[:, None]
        )  # formula: (x - mean) / std

        bad_in_window = np.where(np.any(z_scores > threshold, axis=1))[0]
        for i in bad_in_window:
            bad_channels.add(raw_data.ch_names[i])

    return list(bad_channels)


def get_epochs_from_events(data, event_str, min_reaction_s=None):
    """
    Epoch the given data based on the event_str. Exclude to fast reactions if we are in the case of a deviant event.

    Args:
        data (mne.raw): EEG Data to epoch
        event_str (string): String to determine for which event to epoch to
        min_reaction_s (float, optional): Time in seconds for the minimal reaction time at deviant events. Defaults to None.

    Returns:
        mne.Epoch: Epoch of the data for the given event
    """
    evts, evts_dict = mne.events_from_annotations(data)

    # identify deviant events
    deviant_keys = [e for e in evts_dict.keys() if e.endswith(event_str)]
    correct_keys = [e for e in evts_dict.keys() if "STATUS:Correct - " in e]

    # construct a dictionary of deviant events
    evts_dict_stim = {}
    for key in deviant_keys:
        evts_dict_stim[key] = evts_dict[key]

    data.info.normalize_proj()
    # reject threshold
    reject = dict(eeg=0.0004)  # in V

    epochs = mne.Epochs(
        data,
        evts,
        evts_dict_stim,
        tmin=-0.4,
        tmax=1.6,
        baseline=(-0.4, 0),
        preload=True,
        reject=reject,
    )

    if min_reaction_s:
        # reconstruct which epochs where droped
        rejected_idx = []
        epoch_idx = 0

        for i, log in enumerate(epochs.drop_log):
            if len(log) == 0:
                epoch_idx += 1
            elif not log[0] == "IGNORED":
                rejected_idx.append(epoch_idx)
                epoch_idx += 1

        # get the labels, which where used for epoching
        d_evts = evts[evts[:, 2] == evts_dict_stim[deviant_keys[0]]]

        for key in deviant_keys[1:]:
            d_evts = np.concatenate((d_evts, evts[evts[:, 2] == evts_dict_stim[key]]))

        # sort events by time, to fit indexes to the indices in epochs
        d_evts = sorted(d_evts, key=lambda x: x[0])

        reaction_times = []
        i = 0
        for epoch_idx, d_evt in enumerate(d_evts):
            if not epoch_idx in rejected_idx:
                # find the closest correct event
                c_evts = evts[
                    np.isin(evts[:, 2], [evts_dict[key] for key in correct_keys])
                    & (evts[:, 0] > d_evt[0])
                ]
                if len(c_evts) > 0:
                    # calc the reaction time
                    reaction_time = (c_evts[0][0] - d_evt[0]) / data.info[
                        "sfreq"
                    ]  # convert to ms
                    reaction_times.append((reaction_time, key, i))
                    i += 1

        # Filter epochs based on reaction time
        valid_epochs = [i for (rt, _, i) in reaction_times if min_reaction_s <= rt]
        print(
            f"Filtered {len(valid_epochs)} epochs out of {len(epochs)} based on reaction time threshold"
        )
        epochs = epochs[valid_epochs]

    return epochs


def interpolate_bads_and_merge(blocks):
    """interpolates bad channels and merges the blocks using mne

    Args:
        blocks (list): List of mne.io.Raw objects

    Returns:
        mne.io.Raw: The merged raw data, interpolated separately for each block
    """
    # Interpolate bad channels
    for block in blocks:
        block.interpolate_bads()

    # Merge the blocks
    raw = mne.concatenate_raws(blocks)

    return raw


def set_bad_channels_from_json(blocks, bad_json):
    """sets the bad channels for each block from the given json

    Args:
        blocks (list): List of mne.io.Raw objects
        bad_json (dict): Dictionary containing the bad channels for each block for each subject

    Returns:
        blocks: List of mne.io.Raw objects with bad channels set
    """
    for block in blocks:
        # Set bad channels
        block.info["bads"] = bad_json[f"sub-{SUBJECT}"][f"{blocks.index(block)+1}"]

    return blocks


def apply_autoreject_info(epochs, autoreject_info):
    """applies the rejection thresholds and drops bad epochs from stored data

    Args:
        epochs (mne.Epochs): The epochs to apply the autoreject info to
        autoreject_info (dict): The autoreject info to apply

    Returns:
        mne.Epochs: The epochs with the autoreject info applied
    """
    threshes = autoreject_info["threshes"]
    reject_log = autoreject_info["reject_log"]

    # apply learned thresholds
    epochs.drop_bad(reject=threshes)

    # mark and drop bad epochs
    for i, is_bad in enumerate(reject_log):
        if is_bad:
            epochs.drop(i)

    return epochs


def epoch_rejection(epochs, shape):
    """alternative rejection: rejects epochs based on hard criteria

    Args:
        epochs (mne.Epochs): The epochs to reject
        shape (tuple): The shape of the epochs

    Returns:
        np.array: The rejected epochs
    """
    std_concat = np.std(epochs)

    # preallocate memory for the rejected epochs
    epochs_concat_removed = np.zeros(shape=shape)
    idx = 0  # index to keep track of the position in the pre-allocated array

    print("len prior: ", len(epochs))

    for epoch in epochs:
        for channel in epoch:
            channel_max = np.max(np.abs(channel))
            std_channel = np.std(channel)

            # apply the rejection criteria
            if (
                channel_max < (5 * std_concat)
                and channel_max < (250 * 1e-6)
                and channel_max < (5 * std_channel)
            ):
                if idx < shape[0]:  # check to prevent index out of bounds
                    epochs_concat_removed[idx] = channel
                    idx += 1

    # truncate the array to the actual size
    epochs_concat_removed = epochs_concat_removed[:idx]

    print("len after: ", len(epochs_concat_removed))

    return epochs_concat_removed
