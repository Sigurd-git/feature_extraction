import os
import hdf5storage
import numpy as np
import torch
import torchaudio
from general_analysis_code.preprocess import align_time


def generate_deepspeech2_features(
    wav_path, output_root, encoder_deepspeech, n_t, out_sr=100
):
    """
    This function generates deepspeech2 features for a given wav file and saves them in the specified output directory.

    Parameters:
    wav_path (str): The path to the wav file for which features are to be extracted.
    output_root (str): The root directory where the extracted features will be saved.
    encoder_deepspeech (deepspeech_encoder): The pre-trained deepspeech2 model, which is generated by:
        from general_analysis_code.get_model_output import deepspeech_encoder
        encoder_deepspeech = deepspeech_encoder("/scratch/snormanh_lab/Sigurd/feature_extration/code/utils/deepspeech2-pretrained.ckpt", device=device,compile_torch=compile_torch)
    The function creates a directory for the wav file in the output root directory if it doesn't exist.
    It then extracts deepspeech2 features for the wav file and saves each layer of features as a .mat file in the created directory.
    It also generates a meta file for each layer of features with information about the start time of the timestamps and the sample rate.

    There are 1 spectrogram layer, 2 conv layers, 5 lstm layers and 1 fc layer.
    conv_stride = ((2,2), (2,1)), conv_kernel = ((41,11), (21,11)) and conv_padding = ((20,5), (10,5)),
    rnn_input_size=1312, rnn_hidden_size=1024,
    fc_input_size=1024, fc_output_size=29.

    """
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "deepspeech2")
    meta_out_path = os.path.join(output_root, "feature_metadata", "deepspeech2.txt")
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(os.path.dirname(meta_out_path)):
        os.makedirs(os.path.dirname(meta_out_path))

    waveform, sample_rate = torchaudio.load(wav_path)
    t_num_new = int(np.round(waveform.shape[1] / sample_rate * out_sr))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    sr = 50
    # pad 640 before and pad 640 after
    waveform = torch.nn.functional.pad(waveform, (640, 640), "constant", 0)

    df = encoder_deepspeech.extract_deepSpeech_feature(waveform, 16000)
    time_stamp = df.iloc[0]["time_stamp"]
    features = df["activation"]

    t_0 = (time_stamp[0] - 640) / 16000  # the time of first point, unit is second
    for i, feats in enumerate(features):
        feature_variant_out_dir = os.path.join(feature_class_out_dir, f"layer{i}")
        if not os.path.exists(feature_variant_out_dir):
            os.makedirs(feature_variant_out_dir)
        ### align time to start from 0 and make the length to be n_t
        t_length = feats.shape[0]
        t_origin = np.arange(t_length) / sr + t_0
        if n_t is not None:
            t_new = np.arange(n_t) / out_sr
        else:
            t_new = np.arange(t_num_new) / out_sr
        feats = align_time(feats, t_origin, t_new, "t f")
        print(f"Feature {i}: {feats.shape}")
        # save each layer as mat
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
        # save data as mat
        hdf5storage.savemat(out_mat_path, {"features": feats})
        # generate meta file for this layer
        with open(
            os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.txt"), "w"
        ) as f:
            f.write(f"""The shape of this layer is {feats.shape}.""")

    # generate meta file for this feature
    with open(meta_out_path, "w") as f:
        f.write(
            """timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/deepspeech2/<layer>/<stimuli>.txt as (n_time,n_feature). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr.
There are 1 spectrogram layer, 2 conv layers, 5 lstm layers and 1 fc layer.
conv_stride = ((2,2), (2,1)), conv_kernel = ((41,11), (21,11)) and conv_padding = ((20,5), (10,5)),
rnn_input_size=1312, rnn_hidden_size=1024,
fc_input_size=1024, fc_output_size=29.
"""
        )