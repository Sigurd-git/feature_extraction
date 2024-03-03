import numpy as np
import scipy.io
import torch
import torchaudio
from general_analysis_code.get_model_output import deepspeech_encoder, glove_encoder
from utils.feature_extraction import (
    generate_deepspeech2_features,
    generate_glove_feature,
    generate_wav2vec2_features,
)

######personalize parameters
device = torch.device("cuda:3")
compile_torch = True

glove_fast = "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/glove_encoder.pkl"  # The glove model took too long to load the token dictionary into memory, so I saved an instantiated encoder using dill. This way, it doesn't need to be instantiated from scratch every time.
encoder_glove = glove_encoder(glove_fast)
encoder_deepspeech = deepspeech_encoder(
    "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/deepspeech2-pretrained.ckpt",
    device=device,
    compile_torch=compile_torch,
)

# HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE

# HUBERT_model = HUBERT_bundle.get_model().to(device)

wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_BASE  # or WAV2VEC2_BASE

wav2vec2_model = wav2vec2_bundle.get_model().to(device)

if compile_torch:
    # HUBERT_model = torch.compile(HUBERT_model)
    wav2vec2_model = torch.compile(wav2vec2_model)
# eng = matlab.engine.start_matlab()
# eng.addpath(eng.genpath('/home/gliao2/samlab_Sigurd/feature_extration/code/utils'))
sr = 100
####################speech-long-TCI####################
output_root = "/scratch/snormanh_lab/shared/projects/speech-long-TCI-section/analysis/"
phon_word_mat = scipy.io.loadmat(
    "/home/gliao2/samlab_Sigurd/encodingmodel/data/speech-long-TCI/formatted-stim-annotations_all-conds.mat"
)["Y"]
# load data
data = scipy.io.loadmat(
    "/scratch/snormanh_lab/shared/Sigurd_and_Dana/formatted-data_all-conds_7subjs_v3.mat"
)["D"]

rates = ["fast", "slow"]
naturals = ["natural", "random"]
sections = list(range(4))


def get_all_phoneme_word_labels(phon_word_mat, rates, naturals, sections):
    all_word_labels, all_phoneme_labels = [], []
    for rate_index, rate in enumerate(rates):
        for natural_index, natural in enumerate(naturals):
            for section in sections:
                if natural == "natural":
                    scram = "intact"
                else:
                    scram = "scram"

                word_labels = phon_word_mat[rate][0][0][scram][0, 0]["word"][0][
                    section
                ]["labels"][0][0].reshape(-1)
                word_labels = np.array([label[0] for label in word_labels])
                phoneme_labels = phon_word_mat[rate][0][0][scram][0, 0]["phon"][0][
                    section
                ]["labels"][0][0].reshape(-1)
                phoneme_labels = np.array([label[0] for label in phoneme_labels])

                all_word_labels.append(word_labels)
                all_phoneme_labels.append(phoneme_labels)

    all_word_labels = np.concatenate(all_word_labels)
    all_phoneme_labels = np.concatenate(all_phoneme_labels)
    word_counts = {
        word_label: np.sum(all_word_labels == word_label)
        for word_label in np.unique(all_word_labels)
    }
    all_word_labels = np.unique(all_word_labels)

    phoneme_counts = {
        phoneme_label: np.sum(all_phoneme_labels == phoneme_label)
        for phoneme_label in np.unique(all_phoneme_labels)
    }
    all_phoneme_labels = np.unique(all_phoneme_labels)

    return all_word_labels, all_phoneme_labels, word_counts, phoneme_counts


(
    all_word_labels,
    all_phoneme_labels,
    word_counts,
    phoneme_counts,
) = get_all_phoneme_word_labels(phon_word_mat, rates, naturals, sections)
merge_set = ["AW", "CH", "TH", "ZH", "OY"]


def get_phoneme_word_onset_offset_label(phon_word_mat, rate, section, scram):
    phoneme_labels = phon_word_mat[rate][0][0][scram][0, 0]["phon"][0][section][
        "labels"
    ][0][0].reshape(-1)
    phoneme_labels = np.array([label[0] for label in phoneme_labels])
    phoneme_onsets = phon_word_mat[rate][0][0][scram][0, 0]["phon"][0][section]["ons"][
        0
    ][0].reshape(-1)
    phoneme_offsets = phon_word_mat[rate][0][0][scram][0, 0]["phon"][0][section][
        "offs"
    ][0][0].reshape(-1)
    word_labels = phon_word_mat[rate][0][0][scram][0, 0]["word"][0][section]["labels"][
        0
    ][0].reshape(-1)
    word_labels = np.array([label[0] for label in word_labels])
    word_onsets = phon_word_mat[rate][0][0][scram][0, 0]["word"][0][section]["ons"][0][
        0
    ].reshape(-1)
    word_offsets = phon_word_mat[rate][0][0][scram][0, 0]["word"][0][section]["offs"][
        0
    ][0].reshape(-1)

    return (
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        word_labels,
        word_onsets,
        word_offsets,
    )


for rate_index, rate in enumerate(rates):
    for natural_index, natural in enumerate(naturals):
        for section in sections:
            # get coch features
            if natural == "natural":
                seg = "seg180"
                scram = "intact"
            else:
                seg = "seg4"
                scram = "scram"

            if rate == "fast" and natural == "natural":
                stim = 1
            elif rate == "fast" and natural == "random":
                stim = 2
            elif rate == "slow" and natural == "natural":
                stim = 3
            elif rate == "slow" and natural == "random":
                stim = 4

            stim_name = f"stim{stim}_section{section+1}_{rate}-{seg}"

            wav_path = f"/home/gliao2/samlab_Sigurd/encodingmodel/data/speech-long-TCI/fastslow-story1/{stim_name}.wav"

            # get phoneme features and word timing
            (
                phoneme_labels,
                phoneme_onsets,
                phoneme_offsets,
                word_labels,
                word_onsets,
                word_offsets,
            ) = get_phoneme_word_onset_offset_label(phon_word_mat, rate, section, scram)

            y = data[rate_index, natural_index, section, 0]
            n_t = y.shape[0]
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
            # generate_phoneme_features(output_root, wav_path, phoneme_labels, phoneme_onsets, phoneme_offsets, n_t, phoneme_counts=None, low_prob=None, merge_set=merge_set, attribute=False)
            # generate_cochleagram(eng, wav_path, output_root, n_t)
            # generate_HUBERT_features(device, HUBERT_model, wav_path, output_root,n_t,out_sr=100)
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
            generate_deepspeech2_features(
                wav_path, output_root, encoder_deepspeech, n_t
            )
            generate_wav2vec2_features(
                device, wav2vec2_model, wav_path, output_root, n_t
            )
