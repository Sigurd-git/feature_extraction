import numpy as np
import hdf5storage
import os
import scipy.io

from utils.feature_extraction import generate_pc


####################syllable_invariances####################
pc = 100
feature_names = ["cochleagram"]
dnn_features = ["hubert"]
dnn_layer_nums = [19]

mat = scipy.io.loadmat(
    "/scratch/snormanh_lab/shared/Sigurd_and_Dana/syllable-invariances-data.mat"
)
stim_names = np.concatenate(np.concatenate(mat["stim_names"]))
for feature_name in feature_names:
    for version in range(2):
        wav_features, wav_noext_names = [], []
        if version == 0:
            output_root = "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v1/analysis"
            wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v5/final-stims/wav"
        else:
            output_root = "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v2/analysis"
            wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"

        for stim_index, stim_name in enumerate(stim_names):
            rate = stim_name.split("-")[1]
            if rate == "durmatched":
                continue

            feature_path = (
                f"{output_root}/features/{feature_name}/original/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]

            wav_features.append(feature)
            wav_noext_names.append(f"{stim_name}")

        pca_pipeline = generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            wav_noext_names,
            demean=True,
            std=False,
        )
        generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            wav_noext_names,
            demean=True,
            std=False,
            pca_pipeline=pca_pipeline,
        )

# DNN features

# for feature_name, layer_num in zip(dnn_features, dnn_layer_nums):
#     for layer in range(layer_num):
#         for version in range(2):
#             wav_features, wav_noext_names = [], []
#             if version == 0:
#                 output_root = "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v1/analysis"
#                 wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v5/final-stims/wav"
#             else:
#                 output_root = "/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v2/analysis"
#                 wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"

#             for stim_index, stim_name in enumerate(stim_names):
#                 rate = stim_name.split("-")[1]
#                 if rate == "durmatched":
#                     continue
#                 feature_path = f"{output_root}/features/{feature_name}/layer{layer}/{stim_name}.mat"
#                 feature = hdf5storage.loadmat(feature_path)["features"]

#                 wav_features.append(feature)
#                 wav_noext_names.append(f"{stim_name}")

#             pca_pipeline = generate_pc(
#                 wav_features,
#                 pc,
#                 output_root,
#                 feature_name,
#                 wav_noext_names,
#                 demean=True,
#                 std=False,
#                 variant=f"layer{layer}",
#             )
#             generate_pc(
#                 wav_features,
#                 pc,
#                 output_root,
#                 feature_name,
#                 wav_noext_names,
#                 demean=True,
#                 std=False,
#                 variant=f"layer{layer}",
#                 pca_pipeline=pca_pipeline,
#             )
