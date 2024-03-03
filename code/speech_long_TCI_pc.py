
import numpy as np
import hdf5storage
import os

from utils.feature_extraction import generate_pc


####################speech-long-TCI####################
output_root = "analysis/speech-long-TCI"
pc=50

rates = ["fast", "slow"]
naturals = ["natural", "random"]
sections = list(range(4))
feature_names = ['cochleagrams','glove_features']
dnn_features = ['deepspeech2_features','hubert_features','wav2vec2_features']
dnn_layer_nums = [8,12,12]

for feature_name in feature_names:
    wav_features, wav_noext_names = [], []
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
                feature_path = f"{output_root}/{feature_name}/{stim_name}/{stim_name}.mat"
                feature = hdf5storage.loadmat(feature_path)['features']

                wav_features.append(feature)
                wav_noext_names.append(f"{stim_name}")


    generate_pc(wav_features, pc, output_root, feature_name, wav_noext_names, demean=True,std=False)



# DNN features

for feature_name,layer_num in zip(dnn_features,dnn_layer_nums):
    for layer in range(layer_num):
        wav_features, wav_noext_names = [], []
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
                    feature_path = f"{output_root}/{feature_name}/{stim_name}/{stim_name}_layer{layer}.mat"
                    feature = hdf5storage.loadmat(feature_path)['features']

                    wav_features.append(feature)
                    wav_noext_names.append(f"{stim_name}")
        
        generate_pc(wav_features, pc, output_root, feature_name, wav_noext_names, demean=True,std=False,layer=layer)