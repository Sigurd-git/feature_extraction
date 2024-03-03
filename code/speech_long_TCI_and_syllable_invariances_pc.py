
import numpy as np
import hdf5storage
import os
import scipy.io

from utils.feature_extraction import generate_pc


####################syllable_invariances####################
pc=50
# feature_names = ['cochleagrams','glove_features']
# dnn_features = ['deepspeech2_features','hubert_features','wav2vec2_features']
dnn_features = ['hubert']
dnn_layer_nums = [19]
rates = ["fast", "slow"]
naturals = ["natural", "random"]
sections = list(range(4))
mat = scipy.io.loadmat(
    "/scratch/snormanh_lab/shared/Sigurd_and_Dana/syllable-invariances-data.mat"
)
stim_names = np.concatenate(np.concatenate(mat["stim_names"]))
# for feature_name in feature_names:
    
#     for version in range(2):
#         wav_features, wav_noext_names = [], []
#         if version == 0:
#             output_root = "analysis/syllable-invariances/version1"
#             wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v5/final-stims/wav"
#         else:
#             output_root = "analysis/syllable-invariances/version2"
#             wav_dir = "/scratch/snormanh_lab/dboebing/syllable-invariances/stimuli/harvard-ieee_v8/final-stims/wav"

#         for stim_index, stim_name in enumerate(stim_names):
#             rate = stim_name.split("-")[1]
#             if rate == "durmatched":
#                 continue

#             feature_path = f"{output_root}/{feature_name}/{stim_name}/{stim_name}.mat"
#             feature = hdf5storage.loadmat(feature_path)['features']

#             wav_features.append(feature)
#             wav_noext_names.append(f"{stim_name}")


#         generate_pc(wav_features, pc, output_root, feature_name, wav_noext_names, demean=True,std=False)

# DNN features
output_root_syl = "/scratch/snormanh_lab/shared/projects/syllable-invariances_v{}/analysis"
output_root_tci = "/scratch/snormanh_lab/shared/projects/speech-long-TCI-section/analysis"
for feature_name,layer_num in zip(dnn_features,dnn_layer_nums):
    for layer in range(layer_num):
        for version in (1,2):
            wav_features, wav_noext_names,output_roots = [], [], []
            for stim_index, stim_name in enumerate(stim_names):
                rate = stim_name.split("-")[1]
                if rate == "durmatched":
                    continue
                output_root_syl_version = output_root_syl.format(version)
                feature_path = f"{output_root_syl_version}/features/{feature_name}/layer{layer}/{stim_name}.mat"
                feature = hdf5storage.loadmat(feature_path)['features']

                wav_features.append(feature)
                wav_noext_names.append(f"{stim_name}")
                output_roots.append(output_root_syl_version)
            pca_pipeline = generate_pc(wav_features, pc, output_roots, feature_name, wav_noext_names, demean=True,std=False)
            generate_pc(wav_features, pc, output_roots, feature_name, wav_noext_names, demean=True,std=False,layer=layer,pca_pipeline=pca_pipeline)
            wav_features, wav_noext_names,output_roots = [], [], []
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
                        match (rate, natural):
                            case ("fast", "natural"):
                                stim = 1
                            case ("fast", "random"):
                                stim = 2
                            case ("slow", "natural"):
                                stim = 3
                            case ("slow", "random"):
                                stim = 4


                        stim_name = f"stim{stim}_section{section+1}_{rate}-{seg}"
                        feature_path = f"{output_root_tci}/features/{feature_name}/layer{layer}/{stim_name}.mat"
                        feature = hdf5storage.loadmat(feature_path)['features']

                        wav_features.append(feature)
                        wav_noext_names.append(f"{stim_name}")
                        output_roots.append(output_root_tci)

            generate_pc(wav_features, pc, output_roots, feature_name, wav_noext_names, demean=True,std=False,layer=layer,appendix='_weight-from-syl',pca_pipeline=pca_pipeline)