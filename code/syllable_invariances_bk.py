import os
import re

import hdf5storage
import numpy as np
import scipy.io
import torch
# import torchaudio
import whisper
# from general_analysis_code.get_model_output import deepspeech_encoder, glove_encoder
from utils.feature_extraction import (
    # generate_deepspeech2_features,
    # generate_glove_feature,
    # generate_wav2vec2_features,
    generate_whisper_features
)

######personalize parameters
device = torch.device("cuda:0")
compile_torch = True
sr = 100
# glove_fast = "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/glove_encoder.pkl"  # The glove model took too long to load the token dictionary into memory, so I saved an instantiated encoder using dill. This way, it doesn't need to be instantiated from scratch every time.
# encoder_glove = glove_encoder(glove_fast)
# encoder_deepspeech = deepspeech_encoder(
#     "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/deepspeech2-pretrained.ckpt",
#     device=device,
#     compile_torch=compile_torch,
# )

# HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE

# HUBERT_model = HUBERT_bundle.get_model().to(device)

# wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_BASE  # or WAV2VEC2_BASE

# wav2vec2_model = wav2vec2_bundle.get_model().to(device)

whisper_model = whisper.load_model("large-v2",device=device)


if compile_torch:
    whisper_model = torch.compile(whisper_model)
    # HUBERT_model = torch.compile(HUBERT_model)
    # wav2vec2_model = torch.compile(wav2vec2_model)
# eng = matlab.engine.start_matlab()
# eng.addpath(eng.genpath('/home/gliao2/samlab_Sigurd/feature_extration/code/utils'))

####################syllable-invariances####################

mat = scipy.io.loadmat(
    "/scratch/snormanh_lab/shared/Sigurd_and_Dana/syllable-invariances-data.mat"
)
merge_set = ["AW", "CH", "TH", "ZH", "OY"]
stim_names = np.concatenate(np.concatenate(mat["stim_names"]))
output_root = "/scratch/snormanh_lab/shared/projects/syllable-invariances_v{}/analysis"
wav_dir = "/scratch/snormanh_lab/shared/projects/syllable-invariances_v{}/stimuli/stimulus_audio"


def get_phoneme_word_onset_offset_label(stim_name):
    phon_word_mat = hdf5storage.loadmat(
        f"/scratch/snormanh_lab/shared/Sigurd/encodingmodel/data/syllable-invariances/{stim_name}.mat"
    )
    phoneme_labels = phon_word_mat["phon_labels"]
    phoneme_labels = np.array([x[0].squeeze() for x in phoneme_labels])
    phoneme_onsets = phon_word_mat["phon_onsets"]
    phoneme_durations = phon_word_mat["phon_durations"]
    phoneme_offsets = phoneme_onsets + phoneme_durations

    word_labels = phon_word_mat["word_labels"]
    word_labels = np.array([x[0].squeeze() for x in word_labels])
    word_onsets = phon_word_mat["word_onsets"]
    word_durations = phon_word_mat["word_durations"]
    word_offsets = word_onsets + word_durations

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
    phoneme_labels = np.array(phoneme_labels)

    word_counts = {
        word_label: np.sum(word_labels == word_label)
        for word_label in np.unique(word_labels)
    }
    phoneme_counts = {
        phoneme_label: np.sum(phoneme_labels == phoneme_label)
        for phoneme_label in np.unique(phoneme_labels)
    }
    # find out phoneme_label which count is less than 10
    phoneme_counts_less_than_10 = [
        phoneme_label
        for phoneme_label in phoneme_counts
        if phoneme_counts[phoneme_label] < 10
    ]
    return (
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        word_labels,
        word_onsets,
        word_offsets,
        word_counts,
        phoneme_counts,
        np.unique(word_labels),
    )


for version in (1, 2):
    for stim_index, stim_name in enumerate(stim_names):
        if "durmatch" in stim_name:
            continue
        # get the ys
        y = mat["data"][0, 0][0, stim_index][400:-400]  # time x rep x electrode
        n_t = y.shape[0]
        # get phoneme features and word timing
        (
            phoneme_labels,
            phoneme_onsets,
            phoneme_offsets,
            word_labels,
            word_onsets,
            word_offsets,
            word_counts,
            phoneme_counts,
            all_word_labels,
        ) = get_phoneme_word_onset_offset_label(stim_name)
        wav_path = os.path.join(wav_dir.format(version), f"{stim_name}.wav")
        # generate_feature_set_for_one_stim(
        # device,
        # encoder_glove,
        # encoder_deepspeech,
        # HUBERT_model,
        # wav2vec2_model,
        # eng,
        # wav_path,
        # output_root,
        # n_t, # number of time steps in response
        # word_labels,
        # word_onsets,
        # word_offsets,
        # phoneme_labels,
        # phoneme_onsets,
        # phoneme_offsets,
        # word_counts,
        # all_word_labels,
        # phoneme_counts,
        # sr=100,
        # word_low_prob=None,
        # phoneme_low_prob=None,
        # merge_set=None,
        # attribute=False)
        # generate_phoneme_features(output_root.format(version), wav_path, phoneme_labels, phoneme_onsets, phoneme_offsets, n_t, phoneme_counts=None, low_prob=None, merge_set=merge_set, attribute=False)
        # generate_cochleagram(eng, wav_path, output_root.format(version), n_t)
        # generate_HUBERT_features(device, HUBERT_model, wav_path, output_root.format(version),n_t,out_sr=100)
        # generate_glove_feature(
        #     encoder_glove,
        #     output_root,
        #     wav_path,
        #     word_labels,
        #     word_onsets,
        #     word_offsets,
        #     sr,
        #     n_t,
        # )
        # generate_deepspeech2_features(wav_path, output_root, encoder_deepspeech, n_t)
        # generate_wav2vec2_features(device, wav2vec2_model, wav_path, output_root, n_t)
        generate_whisper_features(device, whisper_model, wav_path, output_root, n_t)
