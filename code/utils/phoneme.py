import os
import re

import hdf5storage
import numpy as np
import pandas as pd
def generate_phoneme_features(
    output_root,
    wav_path,
    phoneme_labels,
    phoneme_onsets,
    phoneme_offsets,
    n_t,
    phoneme_counts=None,
    low_prob=None,
    merge_set=None,
    attribute=False,
):
    """
    Generate phoneme features from a given wav file and its corresponding phoneme labels.

    Args:
        output_root (str): The root directory for the output files.
        wav_path (str): The path to the wav file.
        phoneme_labels (numpy.ndarray): An array of phoneme labels.
        phoneme_onsets (numpy.ndarray): An array of phoneme onsets.
        phoneme_offsets (numpy.ndarray): An array of phoneme offsets.
        n_t (int): The number of time steps.
        phoneme_counts (dict, optional): A dictionary of phoneme counts. Defaults to None.
        low_prob (int, optional): The threshold for low probability phonemes. Defaults to None.
        merge_set (list, optional): A list of phonemes to merge. Defaults to None.
        attribute (bool, optional): Whether to use attribute-based phonemes. Defaults to False.

    Returns:
        None
    """
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "phoneme")
    meta_out_dir = os.path.join(output_root, "feature_metadata")
    feature_variant_discrete_dir = os.path.join(feature_class_out_dir, "discrete")
    out_csv_path = os.path.join(feature_variant_discrete_dir, f"{wav_name_no_ext}.csv")

    if low_prob:
        feature_variant_dir = os.path.join(feature_class_out_dir, "onehot_lowprob")
    elif merge_set is not None:
        feature_variant_dir = os.path.join(
            feature_class_out_dir, f"onehot_merge_{'-'.join(merge_set)}"
        )
    elif attribute:
        feature_variant_dir = os.path.join(feature_class_out_dir, "onehot_attribute")
    else:
        feature_variant_dir = os.path.join(feature_class_out_dir, "onehot")

    if not os.path.exists(feature_variant_dir):
        os.makedirs(feature_variant_dir)
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(meta_out_dir):
        os.makedirs(meta_out_dir)
    if not os.path.exists(feature_variant_discrete_dir):
        os.makedirs(feature_variant_discrete_dir)

    out_mat_path = os.path.join(feature_variant_dir, f"{wav_name_no_ext}.mat")

    all_phoneme_labels = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    ]

    # remove spn from phoneme_labels
    phoneme_indexes = np.array(
        [i for i, phoneme_label in enumerate(phoneme_labels) if phoneme_label != "spn"]
    )
    phoneme_labels = phoneme_labels[phoneme_indexes]
    phoneme_onsets = phoneme_onsets[phoneme_indexes]
    phoneme_offsets = phoneme_offsets[phoneme_indexes]

    # capitilize phoneme_labels
    phoneme_labels = np.array([label.upper() for label in phoneme_labels])
    # romove numbers from phoneme_labels
    phoneme_labels = [re.sub(r"\d+", "", phoneme) for phoneme in phoneme_labels]

    # save phoneme as 3 column matrix. Label, onset, offset
    phoneme_onsets = phoneme_onsets.reshape(-1)
    phoneme_offsets = phoneme_offsets.reshape(-1)
    phoneme_labels = np.array(phoneme_labels, dtype="<U7")

    # create dataframe
    df = pd.DataFrame(
        {"label": phoneme_labels, "onset": phoneme_onsets, "offset": phoneme_offsets}
    )
    df.to_csv(out_csv_path, index=False)
    if low_prob is not None:
        assert isinstance(low_prob, int), "low_prob must be an integer."
        assert low_prob > 0, "low_prob must be larger than 0."
        assert (
            phoneme_counts is not None
        ), "phoneme_counts must be provided if low_prob is not None."
        assert merge_set is None, "merge_set and low_prob cannot both be existed."
        assert attribute is False, "attribute and low_prob cannot both be True."
        # rename phonemes labels that occur fewer to lowprob
        phoneme_indexes = np.array(
            [
                i
                for i, phoneme_label in enumerate(phoneme_labels)
                if phoneme_counts[phoneme_label] < 10
            ]
        )
        if len(phoneme_indexes) > 0:
            phoneme_labels = np.array(phoneme_labels, dtype="<U7")
            phoneme_labels[phoneme_indexes] = "LOWPROB"
        all_phoneme_labels.append("LOWPROB")
    if merge_set is not None:
        assert (
            phoneme_counts is None
        ), "phoneme_counts must not be provided if merge_set is not None."
        assert low_prob is None, "merge_set and low_prob cannot both be existed."
        assert attribute is False, "attribute and merge_set cannot both be existed."

        phoneme_indexes = np.array(
            [
                i
                for i, phoneme_label in enumerate(phoneme_labels)
                if phoneme_label in merge_set
            ]
        )
        if len(phoneme_indexes) > 0:
            phoneme_labels = np.array(phoneme_labels, dtype="<U7")
            phoneme_labels[phoneme_indexes] = "MERGED"
        all_phoneme_labels.append("MERGED")

        # remove merged phonemes from all_phoneme_labels
        for phoneme in merge_set:
            all_phoneme_labels.remove(phoneme)

    if attribute:
        assert (
            phoneme_counts is None
        ), "phoneme_counts must not be provided if attribute is True."
        assert low_prob is None, "attribute and low_prob cannot both be existed."
        assert merge_set is None, "attribute and merge_set cannot both be existed."

        all_phoneme_labels = attribute2phoneme(None, "list")
        tmp_phoneme_labels = []
        tmp_phoneme_onsets = []
        tmp_phoneme_offsets = []
        for phoneme_labels, phoneme_onsets, phoneme_offsets in zip(
            phoneme_labels, phoneme_onsets, phoneme_offsets
        ):
            tmp_phoneme_labels.extend(phoneme2attribute(phoneme_labels))
            tmp_phoneme_onsets.extend(
                [phoneme_onsets] * len(phoneme2attribute(phoneme_labels))
            )
            tmp_phoneme_offsets.extend(
                [phoneme_offsets] * len(phoneme2attribute(phoneme_labels))
            )
        phoneme_labels = np.array(tmp_phoneme_labels, dtype="<U11")
        phoneme_onsets = np.array(tmp_phoneme_onsets)
        phoneme_offsets = np.array(tmp_phoneme_offsets)
    phoneme_features = generate_onehot_features(
        all_phoneme_labels, phoneme_labels, phoneme_onsets, phoneme_offsets, n_t
    )

    # save data as mat
    hdf5storage.savemat(out_mat_path, {"features": phoneme_features})
    # generate meta file for phoneme onehot feature
    with open(os.path.join(feature_variant_dir, f"{wav_name_no_ext}.txt"), "w") as f:
        if low_prob is not None:
            f.write(
                "Lowprob phonemes which appear less than 10 are merged and renamed to 'lowprob'."
            )
        f.write(f"The shape of this feature is {phoneme_features.shape}.")

    # generate meta file for discrete phoneme
    with open(os.path.join(meta_out_dir, "phoneme_discrete.txt"), "w") as f:
        f.write(
            """The 3 columns are: phoneme label, phoneme onset, phoneme offset for each phoneme instance. 
The 39 phonemes we are using are: AA, AE, AH, AO, AW, AY, B, CH, D, DH, EH, ER, EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S, SH, T, TH, UH, UW, V, W, Y, Z, ZH."""
        )

    # generate meta file for phoneme onehot feature
    with open(os.path.join(meta_out_dir, "phoneme_onehot.txt"), "w") as f:
        f.write(
            """The 39 phonemes we are using for this feature are: AA, AE, AH, AO, AW, AY, B, CH, D, DH, EH, ER, EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S, SH, T, TH, UH, UW, V, W, Y, Z, ZH.
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/phoneme/onehot_lowprob/<stimuli>.txt as (n_time,n_phoneme). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
        )
    # generate meta file for phoneme onehot low_prob feature
    if low_prob:
        with open(os.path.join(meta_out_dir, "phoneme_onehot_lowprob.txt"), "w") as f:
            f.write(
                f"""Lowprob phonemes which appear less than 10 are merged and renamed to 'LOWPROB'. The actual label we are using are {all_phoneme_labels}.
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/phoneme/onehot_lowprob/<stimuli>.txt as (n_time,n_phoneme). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
            )
    # generate meta file for phoneme onehot merge feature
    if merge_set is not None:
        with open(os.path.join(meta_out_dir, "phoneme_onehot_merge.txt"), "w") as f:
            f.write(
                f"""Phonemes {merge_set} are merged and renamed to 'merged'. The actual label we are using are {all_phoneme_labels}.
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/phoneme/onehot_merge_{'-'.join(merge_set)}/<stimuli>.txt as (n_time,n_phoneme). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
            )

    # generate meta file for phoneme onehot attribute feature
    if attribute:
        with open(os.path.join(meta_out_dir, "phoneme_onehot_attribute.txt"), "w") as f:
            f.write(
                f"""Phonemes are reclassified according to their attributes. The actual label we are using are {all_phoneme_labels}.
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/phoneme/onehot_attribute/<stimuli>.txt as (n_time,n_phoneme). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
            )
