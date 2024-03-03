import os
import re

import hdf5storage
import numpy as np
import scipy.io
import torch
import torchaudio
import whisper
from general_analysis_code.get_model_output import deepspeech_encoder, glove_encoder
from utils.feature_extraction import (
    generate_deepspeech2_features,
    generate_glove_feature,
    generate_wav2vec2_features,
    generate_whisper_features,
    generate_HUBERT_features,
    generate_CLAP_features,
    generate_pc,
)


######personalize parameters
device = torch.device("cuda:0")
# compile_torch = True
# sr = 100
# glove_fast = "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/glove_encoder.pkl"  # The glove model took too long to load the token dictionary into memory, so I saved an instantiated encoder using dill. This way, it doesn't need to be instantiated from scratch every time.
# encoder_glove = glove_encoder(glove_fast)
# encoder_deepspeech = deepspeech_encoder(
#     "/home/gliao2/samlab_Sigurd/feature_extration/code/utils/deepspeech2-pretrained.ckpt",
#     device=device,
#     compile_torch=compile_torch,
# )

HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE

HUBERT_model = HUBERT_bundle.get_model().to(device)

# wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_BASE  # or WAV2VEC2_BASE

# wav2vec2_model = wav2vec2_bundle.get_model().to(device)

# whisper_model = whisper.load_model("large-v2", device=device)

pc = 100
# if compile_torch:
# whisper_model = torch.compile(whisper_model)
# HUBERT_model = torch.compile(HUBERT_model)
# wav2vec2_model = torch.compile(wav2vec2_model)
# eng = matlab.engine.start_matlab()
# eng.addpath(eng.genpath('/home/gliao2/samlab_Sigurd/feature_extration/code/utils'))

####################intracranial-natsound119####################

mat = hdf5storage.loadmat(
    "/home/gliao2/snormanh_lab_shared/projects/intracranial-natsound119/stimuli/stim_names.mat"
)
stim_names = mat["stim_names"]
output_root = "/scratch/snormanh_lab/shared/projects/intracranial-natsound119/analysis"
wav_dir = "/scratch/snormanh_lab/shared/projects/intracranial-natsound119/stimuli/stimulus_audio"


# syllable_invariances
syl_mat = scipy.io.loadmat(
    "/scratch/snormanh_lab/shared/Sigurd_and_Dana/syllable-invariances-data.mat"
)
syl_stim_names = np.concatenate(np.concatenate(syl_mat["stim_names"]))
syl_stim_names = syl_stim_names.tolist()
# remove dur_matched from syl_stim_names
syl_stim_names = [
    stim_name for stim_name in syl_stim_names if "durmatched" not in stim_name
]

for stim_index, stim_name in enumerate(stim_names):
    wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
    generate_HUBERT_features(device, HUBERT_model, wav_path, output_root, n_t=None)




# ####hubert
# feature_name = "hubert"
# layer_num = 19
# for layer in range(layer_num):
#     wav_features = []
#     for stim_index, stim_name in enumerate(stim_names):
#         wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
#         feature_path = (
#             f"{output_root}/features/{feature_name}/layer{layer}/{stim_name}.mat"
#         )
#         feature = hdf5storage.loadmat(feature_path)["features"]
#         wav_features.append(feature)

#     syl_output_root = (
#         "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v2/analysis"
#     )
#     wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"
#     syl_wav_features = []
#     for stim_index, stim_name in enumerate(syl_stim_names):
#         feature_path = (
#             f"{syl_output_root}/features/{feature_name}/layer{layer}/{stim_name}.mat"
#         )
#         feature = hdf5storage.loadmat(feature_path)["features"]
#         syl_wav_features.append(feature)

#     pca_pipeline = generate_pc(
#         wav_features,
#         pc,
#         output_root,
#         feature_name,
#         stim_names,
#         demean=True,
#         std=False,
#         variant=f"layer{layer}",
#     )

#     # generate_pc(
#     #     wav_features,
#     #     pc,
#     #     output_root,
#     #     feature_name,
#     #     stim_names,
#     #     demean=True,
#     #     std=False,
#     #     variant=f"layer{layer}",
#     #     pca_pipeline=pca_pipeline,
#     # )
#     generate_pc(
#         syl_wav_features,
#         pc,
#         syl_output_root,
#         feature_name,
#         syl_stim_names,
#         demean=True,
#         std=False,
#         variant=f"layer{layer}",
#         pca_pipeline=pca_pipeline,
#     )

# # cochleagram
# feature_name = "cochleagram"
# wav_features = []
# for stim_index, stim_name in enumerate(stim_names):
#     wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
#     feature_path = f"{output_root}/features/{feature_name}/original/{stim_name}.mat"
#     feature = hdf5storage.loadmat(feature_path)["features"]
#     wav_features.append(feature)


# syl_output_root = (
#     "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v2/analysis"
# )
# wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"
# syl_wav_features = []
# for stim_index, stim_name in enumerate(syl_stim_names):
#     feature_path = f"{syl_output_root}/features/{feature_name}/original/{stim_name}.mat"
#     feature = hdf5storage.loadmat(feature_path)["features"]
#     syl_wav_features.append(feature)


# pca_pipeline = generate_pc(
#     wav_features,
#     pc,
#     output_root,
#     feature_name,
#     stim_names,
#     demean=True,
#     std=False,
# )

# generate_pc(
#     wav_features,
#     pc,
#     output_root,
#     feature_name,
#     stim_names,
#     demean=True,
#     std=False,
#     pca_pipeline=pca_pipeline,
# )
# generate_pc(
#     syl_wav_features,
#     pc,
#     syl_output_root,
#     feature_name,
#     syl_stim_names,
#     demean=True,
#     std=False,
#     pca_pipeline=pca_pipeline,
# )

# # spectrotemporal
# feature_name = "spectrotemporal"
# wav_features = []
# for stim_index, stim_name in enumerate(stim_names):
#     wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
#     feature_path = f"{output_root}/features/{feature_name}/modulus/{stim_name}.mat"
#     feature = hdf5storage.loadmat(feature_path)["features"]
#     wav_features.append(feature)


# syl_output_root = (
#     "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v2/analysis"
# )
# wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"
# syl_wav_features = []
# for stim_index, stim_name in enumerate(syl_stim_names):
#     feature_path = f"{syl_output_root}/features/{feature_name}/modulus/{stim_name}.mat"
#     feature = hdf5storage.loadmat(feature_path)["features"]
#     syl_wav_features.append(feature)


# pca_pipeline = generate_pc(
#     wav_features,
#     pc,
#     output_root,
#     feature_name,
#     stim_names,
#     demean=True,
#     std=False,
#     variant="modulus",
# )

# generate_pc(
#     wav_features,
#     pc,
#     output_root,
#     feature_name,
#     stim_names,
#     demean=True,
#     std=False,
#     pca_pipeline=pca_pipeline,
#     variant="modulus",
# )
# generate_pc(
#     syl_wav_features,
#     pc,
#     syl_output_root,
#     feature_name,
#     syl_stim_names,
#     demean=True,
#     std=False,
#     pca_pipeline=pca_pipeline,
#     variant="modulus",
# )
