import os
import re

import hdf5storage
import numpy as np
import pandas as pd

from general_analysis_code.phoneme_class import attribute2phoneme, phoneme2attribute
from general_analysis_code.preprocess import align_time, generate_onehot_features

# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


def generate_non_speech_feature_set_for_one_stim(
    device,
    HUBERT_model,
    wav2vec2_model,
    encoder_deepspeech,
    eng,
    wav_path,
    output_root,
    n_t=None,
):

    # generate wav2vec2 features
    generate_wav2vec2_features(device, wav2vec2_model, wav_path, output_root, n_t)
    # generate deepspeech2 features
    generate_deepspeech2_features(wav_path, output_root, encoder_deepspeech, n_t)


def generate_dnn_feature_set_for_one_stim(
    device, HUBERT_model, wav2vec2_model, encoder_deepspeech, wav_path, output_root, n_t
):
    # generate wav2vec2 features
    generate_wav2vec2_features(device, wav2vec2_model, wav_path, output_root, n_t)

    # generate HUBERT features
    generate_HUBERT_features(device, HUBERT_model, wav_path, output_root, n_t)

    # generate deepspeech2 features
    generate_deepspeech2_features(wav_path, output_root, encoder_deepspeech, n_t)


def generate_feature_set_for_one_stim(
    device,
    encoder_glove,
    encoder_deepspeech,
    HUBERT_model,
    wav2vec2_model,
    eng,
    wav_path,
    output_root,
    n_t,  # number of time steps in response
    word_labels,
    word_onsets,
    word_offsets,
    phoneme_labels,
    phoneme_onsets,
    phoneme_offsets,
    word_counts,
    all_word_labels,
    phoneme_counts,
    sr=100,
    word_low_prob=None,
    phoneme_low_prob=None,
    merge_set=None,
    attribute=False,
):

    # generate glove feature
    generate_glove_feature(
        encoder_glove,
        output_root,
        wav_path,
        word_labels,
        word_onsets,
        word_offsets,
        sr,
        n_t,
    )

    # generate onehot word-identity feature
    generate_word_features(
        output_root,
        wav_path,
        word_labels,
        word_onsets,
        word_offsets,
        n_t,
        word_counts,
        all_word_labels,
        word_low_prob,
    )

    # generate onehot phoneme-identity feature
    generate_phoneme_features(
        output_root,
        wav_path,
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        n_t,
        phoneme_counts=phoneme_counts,
        low_prob=phoneme_low_prob,
        merge_set=merge_set,
        attribute=attribute,
    )

    # generate wav2vec2 features
    generate_wav2vec2_features(device, wav2vec2_model, wav_path, output_root, n_t)

    # generate HUBERT features
    generate_HUBERT_features(device, HUBERT_model, wav_path, output_root, n_t)

    # generate deepspeech2 features
    generate_deepspeech2_features(wav_path, output_root, encoder_deepspeech, n_t)





def generate_notes_features(
    output_root,
    wav_path,
    notes_labels,
    notes_onsets,
    notes_offsets,
    n_t,
    all_notes_labels,
    notes_counts=None,
    low_prob=None,
    merge_set=None,
    onset_feature=False,
    appendix=None,
    discrete=True,
):
    notes_labels = np.array(notes_labels, dtype="<U15")
    all_notes_labels = np.array(all_notes_labels, dtype="<U15")
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "notes")
    meta_out_dir = os.path.join(output_root, "feature_metadata")
    feature_variant_discrete_dir = os.path.join(feature_class_out_dir, "discrete")
    out_csv_path = os.path.join(feature_variant_discrete_dir, f"{wav_name_no_ext}.csv")

    if low_prob:
        feature_variant_dir = os.path.join(
            feature_class_out_dir, f"onehot_lowprob{appendix}"
        )
    elif merge_set is not None:
        feature_variant_dir = os.path.join(
            feature_class_out_dir, f"onehot_merge_{'-'.join(merge_set)}{appendix}"
        )
    else:
        feature_variant_dir = os.path.join(feature_class_out_dir, f"onehot{appendix}")

    if not os.path.exists(feature_variant_dir):
        os.makedirs(feature_variant_dir)
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(meta_out_dir):
        os.makedirs(meta_out_dir)
    if not os.path.exists(feature_variant_discrete_dir):
        os.makedirs(feature_variant_discrete_dir)

    out_mat_path = os.path.join(feature_variant_dir, f"{wav_name_no_ext}.mat")

    # save notes as 3 column matrix. Label, onset, offset
    notes_onsets = notes_onsets.reshape(-1)
    notes_offsets = notes_offsets.reshape(-1)
    notes_labels = np.array(notes_labels)

    if discrete:
        # create dataframe
        df = pd.DataFrame(
            {"label": notes_labels, "onset": notes_onsets, "offset": notes_offsets}
        )
        df.to_csv(out_csv_path, index=False)
        # generate meta file for discrete notes
        with open(os.path.join(meta_out_dir, "discrete.txt"), "w") as f:
            f.write(
                f"""The 3 columns are: notes label, notes onset, notes offset for each notes instance. 
    The {len(all_notes_labels)} notess we are using are: {all_notes_labels}"""
            )
    if low_prob is not None:
        assert isinstance(low_prob, int), "low_prob must be an integer."
        assert low_prob > 0, "low_prob must be larger than 0."
        assert (
            notes_counts is not None
        ), "notes_counts must be provided if low_prob is not None."
        assert merge_set is None, "merge_set and low_prob cannot both be existed."
        # rename notess labels that occur fewer to lowprob
        notes_indexes = np.array(
            [
                i
                for i, notes_label in enumerate(notes_labels)
                if notes_counts[notes_label] < 10
            ]
        )
        if len(notes_indexes) > 0:
            notes_labels[notes_indexes] = "LOWPROB"
        all_notes_labels.append("LOWPROB")
    if merge_set is not None:
        assert (
            notes_counts is None
        ), "notes_counts must not be provided if merge_set is not None."
        assert low_prob is None, "merge_set and low_prob cannot both be existed."

        notes_indexes = np.array(
            [
                i
                for i, notes_label in enumerate(notes_labels)
                if notes_label in merge_set
            ]
        )
        if len(notes_indexes) > 0:
            notes_labels[notes_indexes] = "MERGED"
        all_notes_labels.append("MERGED")

        # remove merged notess from all_notes_labels
        for notes in merge_set:
            all_notes_labels.remove(notes)
    all_notes_labels = np.array(all_notes_labels, dtype=str)
    notes_features = generate_onehot_features(
        all_notes_labels,
        notes_labels,
        notes_onsets,
        notes_offsets,
        n_t,
        onset_feature=onset_feature,
    )

    # save data as mat
    hdf5storage.savemat(out_mat_path, {"features": notes_features})
    # generate meta file for notes onehot feature
    with open(os.path.join(feature_variant_dir, f"{wav_name_no_ext}.txt"), "w") as f:
        if low_prob is not None:
            f.write(
                "Lowprob notess which appear less than 10 are merged and renamed to 'lowprob'."
            )
        f.write(f"The shape of this feature is {notes_features.shape}.")

    if (low_prob is None) & (merge_set is None):
        # generate meta file for notes onehot feature
        with open(os.path.join(meta_out_dir, "notes_onehot.txt"), "w") as f:
            f.write(
                f"""The {len(all_notes_labels)} notess we are using are: {all_notes_labels}"""
            )
    # generate meta file for notes onehot low_prob feature
    if low_prob:
        with open(os.path.join(meta_out_dir, "notes_onehot_lowprob.txt"), "w") as f:
            f.write(
                f"Lowprob notess which appear less than 10 are merged and renamed to 'LOWPROB'. The actual label we are using are {all_notes_labels}"
            )
    # generate meta file for notes onehot merge feature
    if merge_set is not None:
        with open(os.path.join(meta_out_dir, "notes_onehot_merge.txt"), "w") as f:
            f.write(
                f"notess {merge_set} are merged and renamed to 'merged'. The actual label we are using are {all_notes_labels}"
            )


def generate_word_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    n_t,
    word_counts,
    all_word_labels,
    low_prob=None,
):
    """
    This function generates one-hot encoded features for words in a given audio file and saves them as .mat files.

    Parameters:
    output_root (str): The root directory where the output files will be saved.
    word_labels (np.array): An array of word labels.
    word_onsets (np.array): An array of word onset times.
    word_offsets (np.array): An array of word offset times.
    n_t (int): The total number of time frames in the audio file.
    word_counts (dict): A dictionary containing the count of each word label.
    all_word_labels (list): A list of all possible word labels.
    low_prob (bool): If True, words that appear less than 4 times are considered low probability and are renamed to 'lowprob'.

    Returns:
    None. The function saves the one-hot encoded features as .mat files in the specified output directory.
    """
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]

    feature_class_out_dir = os.path.join(output_root, "features", "word")
    meta_out_dir = os.path.join(output_root, "feature_metadata")
    feature_variant_discrete_dir = os.path.join(feature_class_out_dir, "discrete")
    out_csv_path = os.path.join(feature_variant_discrete_dir, f"{wav_name_no_ext}.csv")

    if low_prob:
        feature_variant_dir = os.path.join(feature_class_out_dir, "onehot_lowprob")
    else:
        feature_variant_dir = os.path.join(feature_class_out_dir, "onehot")

    if not os.path.exists(feature_variant_discrete_dir):
        os.makedirs(feature_variant_discrete_dir)
    if not os.path.exists(feature_variant_dir):
        os.makedirs(feature_variant_dir)
    out_mat_path = os.path.join(feature_variant_dir, f"{wav_name_no_ext}.mat")
    if low_prob is not None:
        assert isinstance(low_prob, int), "low_prob must be an integer."
        assert low_prob > 0, "low_prob must be larger than 0."
        # rename phonemes labels that occur fewer than 10 times to lowprob
        word_indexes = np.array(
            [
                i
                for i, word_label in enumerate(word_labels)
                if word_counts[word_label] < low_prob
            ]
        )

        if len(word_indexes) > 0:
            # if the maximun length of word_labels is larger than 7, set the dtype to be maximum, else 7
            if max([len(word) for word in word_labels]) > 7:
                word_labels = np.array(
                    word_labels,
                    dtype="<U" + str(max([len(word) for word in word_labels])),
                )
            else:
                word_labels = np.array(word_labels, dtype="<U7")
            word_labels[word_indexes] = "LOWPROB"
        all_word_labels.append("LOWPROB")
    word_features = generate_onehot_features(
        all_word_labels, word_labels, word_onsets, word_offsets, n_t
    )

    # save data as mat
    hdf5storage.savemat(out_mat_path, {"features": word_features})
    # generate meta file for word onehot feature
    with open(os.path.join(feature_variant_dir, f"{wav_name_no_ext}.txt"), "w") as f:
        if low_prob is not None:
            f.write(
                f"Lowprob words which appear less than {low_prob} are merged and renamed to 'lowprob'."
            )
        f.write(f"The shape of this feature is {word_features.shape}.")

    # save word as 3 column matrix. Label, onset, offset
    word_onsets = word_onsets.reshape(-1)
    word_offsets = word_offsets.reshape(-1)

    # create dataframe
    df = pd.DataFrame(
        {"label": word_labels, "onset": word_onsets, "offset": word_offsets}
    )
    df.to_csv(out_csv_path, index=False)

    # #generate meta file for discrete word
    # with open(os.path.join(feature_variant_discrete_dir, f"{wav_name_no_ext}.txt"), "w") as f:
    #     f.write(f'''The 3 columns are: word label, word onset, word offset for each word instance.
    #             The {len(all_word_labels)} words we are using are: {','.join(all_word_labels)}''')

    # generate meta file for word discrete feature
    with open(os.path.join(meta_out_dir, "word_discrete.txt"), "w") as f:
        f.write(
            f"""The 3 columns are: word label, word onset, word offset for each word instance. 
The {len(all_word_labels)} words we are using are: {','.join(all_word_labels)}. """
        )

    # generate meta file for word onehot feature
    with open(os.path.join(meta_out_dir, "word_onehot.txt"), "w") as f:
        f.write(
            f"""The {len(all_word_labels)} words we are using for this feature are: {','.join(all_word_labels)}. Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/word/onehot/<stimuli>.txt as (n_time,n_word). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
        )

    # generate meta file for word onehot low_prob feature
    if low_prob is not None:
        with open(os.path.join(meta_out_dir, "word_onehot_lowprob.txt"), "w") as f:
            f.write(
                f"""Lowprob words which appear less than {low_prob} are merged and renamed to 'LOWPROB'. The actual label we are using are {all_word_labels}. 
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/word/onehot_lowprob/<stimuli>.txt as (n_time,n_word). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr."""
            )


def generate_glove_feature(
    encoder_glove,
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    sr,
    n_t,
):
    """
    Generate GloVe features for a given audio file and save them as a .mat file.

    Args:
        encoder_glove (EncoderGlove): An instance of EncoderGlove class.
        output_root (str): The root directory to save the generated features.
        wav_path (str): The path to the audio file.
        word_labels (list): A list of word labels.
        word_onsets (list): A list of word onsets.
        word_offsets (list): A list of word offsets.
        sr (int): The sample rate of the audio file.
        n_t (int): The number of time steps.

    Returns:
        None
    """
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "glove")
    meta_out_dir = os.path.join(output_root, "feature_metadata")
    feature_variant_glove_dir = os.path.join(feature_class_out_dir, "original")

    if not os.path.exists(feature_variant_glove_dir):
        os.makedirs(feature_variant_glove_dir)
    if not os.path.exists(meta_out_dir):
        os.makedirs(meta_out_dir)

    out_mat_path = os.path.join(feature_variant_glove_dir, f"{wav_name_no_ext}.mat")
    glove_features = encoder_glove.encode_sequences(
        word_labels, word_onsets, word_offsets, n_t, sr=sr
    )

    # save data as mat
    hdf5storage.savemat(out_mat_path, {"features": glove_features})

    # generate meta file for glove feature
    with open(
        os.path.join(feature_variant_glove_dir, f"{wav_name_no_ext}.txt"), "w"
    ) as f:
        f.write(f"The shape of this feature is {glove_features.shape}.")

    # generate meta file for glove feature
    with open(os.path.join(meta_out_dir, "glove.txt"), "w") as f:
        f.write(
            f"""Timestamps start from 0, sr={sr}Hz. You can find out the shape of each stimulus in features/glove/original/<stimuli>.txt as (n_time,n_phoneme). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr.
The pretrained word vectord of the GloVe model are from: https://nlp.stanford.edu/data/glove.840B.300d.zip.
I split the abbreviation like this dict:

"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"girl's": "girl",
"husband's": "husband",

Then the vector of each abbreviation is the average of the vectors of the words in the abbreviation.
"""
        )


