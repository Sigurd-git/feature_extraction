import os
import hdf5storage
import numpy as np
import torch
import torchaudio
from general_analysis_code.preprocess import align_time
from transformers import AutoProcessor, AutoModel


def generate_CLAP_features(
    device, CLAP_model, wav_path, output_root, n_t=None, out_sr=100
):
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "clap")
    meta_out_path = os.path.join(output_root, "feature_metadata", "clap.txt")
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(os.path.dirname(meta_out_path)):
        os.makedirs(os.path.dirname(meta_out_path))
    waveform, sample_rate = torchaudio.load(wav_path)
    t_num_new = int(np.round(waveform.shape[1] / sample_rate * out_sr))
    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr

    outputs = extract_CLAP(device, CLAP_model, waveform, sample_rate)
    t_0s = [0.01065625, 3.1968031968]
    for i, feats in enumerate(outputs):
        # The start time of output of layer n is: t_0_{n+1} = t_0_{n} + (conv_kernel_{n}-1) / (2*sr_{n})
        # The sr of output of layer n is: sr_{n+1} = sr_{n} / conv_stride_{n}
        t_0 = t_0 + (conv_kernels[i] - 1) / (2 * sr)
        sr = sr / conv_strides[i]
        print(f"t_0_{i}={t_0}, sr_{i}={sr}")
        feature_variant_out_dir = os.path.join(feature_class_out_dir, f"layer{i}")
        if not os.path.exists(feature_variant_out_dir):
            os.makedirs(feature_variant_out_dir)
        feats = feats[0][0].squeeze(0)
        feats = feats.cpu().numpy().transpose()
        ### align time to start from 0 and make the length to be n_t
        t_length = feats.shape[0]
        t_origin = np.arange(t_length) / sr + t_0

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
            f.write(f"""The shape of this conv layer is {feats.shape}.""")

    for j, feats in enumerate(features):
        index = i + j + 1

        feature_variant_out_dir = os.path.join(feature_class_out_dir, f"layer{index}")
        if not os.path.exists(feature_variant_out_dir):
            os.makedirs(feature_variant_out_dir)
        feats = feats.squeeze(0)
        feats = feats.cpu().numpy()
        ### align time to start from 0 and make the length to be n_t
        t_length = feats.shape[0]
        t_origin = np.arange(t_length) / sr + t_0

        feats = align_time(feats, t_origin, t_new, "t f")
        print(f"Feature {index}: {feats.shape}")
        # save each layer as mat
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")

        # save data as mat
        hdf5storage.savemat(out_mat_path, {"features": feats})
        # generate meta file for this layer
        with open(
            os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.txt"), "w"
        ) as f:
            f.write(f"""The shape of this transformer layer is {feats.shape}.""")

    # generate meta file for this feature
    with open(meta_out_path, "w") as f:
        f.write(
            """HuBert features.
Each layer is saved separately as variant. 
Timestamps start from 0 ms, sr=100Hz. You can find out the shape of each stimulus in features/hubert/<layer>/<stimuli>.txt as (n_time,n_feature). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr.
conv_stride = (5, 2, 2, 2, 2, 2, 2) and conv_kernel = (10, 3, 3, 3 ,3 ,2 ,2), no padding for the convolusion layers.
There are 7 conv layers and 12 transformer layers.

"""
        )


def extract_CLAP(device, CLAP_model, waveform, sample_rate):
    # waveform = waveform
    waveform = torch.as_tensor(waveform)
    if sample_rate != 48000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48000).reshape(
            -1
        )
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    inputs = processor(audios=waveform, return_tensors="pt").to(device)
    list(inputs.values())[0].shape
    with torch.inference_mode():
        outputs = CLAP_model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states
    # reshape from 1, n_c, n_f, n_t, to n_c*n_f, n_t
    outputs = [inputs["input_features"].squeeze(0).squeeze(0).T] + [
        activation.reshape(-1, activation.shape[-1]) for activation in activations
    ]
    return outputs
