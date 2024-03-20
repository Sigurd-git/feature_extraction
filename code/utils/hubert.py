import os
import hdf5storage
import numpy as np
import torch
from torch.amp import autocast
import torchaudio
from general_analysis_code.preprocess import align_time
from utils.hook import register_activation_hooks, remove_hooks
from utils.pc import (
    generate_pca_pipeline,
    apply_pca_pipeline,
    generate_pca_pipeline_from_weights,
)
from utils.shared import write_summary, prepare_waveform


def hubert(
    device,
    output_root,
    stim_names,
    wav_dir,
    out_sr=100,
    pc=100,
    time_window=[-1, 1],
    pca_weights_from=None,
    compute_original=True,
    meta_only=False,
    **kwargs,
):
    half = kwargs.get("half", False)
    compile_torch = kwargs.get("compile_torch", True)
    if compute_original:
        # TODO: Use large finetuned model
        HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE
        HUBERT_model = HUBERT_bundle.get_model().to(device)
        if compile_torch:
            HUBERT_model = torch.compile(HUBERT_model)
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            generate_HUBERT_features(
                device,
                HUBERT_model,
                wav_path,
                output_root,
                n_t=None,
                out_sr=out_sr,
                time_window=time_window,
                half=half,
                meta_only=meta_only,
            )

    # compute PC of hubert features
    feature_name = "hubert"
    layer_num = 19
    if pc is not None:
        for layer in range(layer_num):
            wav_features = []
            for stim_index, stim_name in enumerate(stim_names):
                wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
                feature_path = f"{output_root}/features/{feature_name}/layer{layer}/{stim_name}.mat"
                feature = hdf5storage.loadmat(feature_path)["features"]
                wav_features.append(feature)
            print(f"Start computing PCs for {feature_name} layer {layer}")
            if pca_weights_from is not None:
                weights_path = f"{pca_weights_from}/features/{feature_name}/layer{layer}/metadata/pca_weights.mat"
                pca_pipeline = generate_pca_pipeline_from_weights(
                    weights_from=weights_path, pc=pc
                )
            else:
                pca_pipeline = generate_pca_pipeline(
                    wav_features,
                    pc,
                    output_root,
                    feature_name,
                    demean=True,
                    std=False,
                    variant=f"layer{layer}",
                )
            feature_variant_out_dir = apply_pca_pipeline(
                wav_features,
                pc,
                output_root,
                feature_name,
                stim_names,
                variant=f"layer{layer}",
                pca_pipeline=pca_pipeline,
                time_window=time_window,
                sampling_rate=out_sr,
                meta_only=meta_only,
            )


def generate_HUBERT_features(
    device,
    HUBERT_model,
    wav_path,
    output_root,
    n_t,
    out_sr=100,
    time_window=[-1, 1],
    half=False,
    meta_only=False,
):
    """
    This function generates HUBERT features for a given audio file.

    Parameters:
    device (torch.device): The device on which to perform computations (CPU or GPU).
    HUBERT_model (nn.Module): The pre-trained HUBERT model, which is generated by:
        HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE
        HUBERT_model = HUBERT_bundle.get_model().to(device)
    wav_path (str): The path to the .wav file for which to generate features.
    output_root (str): The root directory where the output feature files will be saved.
    n_t (int): The number of time steps.
    out_sr (int): The sample rate of the output features.

    The function saves the generated features as .mat files and also creates a .txt file for each layer of features.

    The start time of output of layer n is: t_0_{n+1} = t_0_{n} + (conv_kernel_{n}-1) / (2*sr_{n})
    The sr of output of layer n is: sr_{n+1} = sr_{n} / conv_stride_{n}

    How I derive the time of first point: The downsampling is achieved through conv1d, so the time axis of features here is 1/320 of the original audio. conv_stride = (5, 2, 2, 2, 2, 2, 2) and conv_kernel = (10, 3, 3, 3 ,3 ,2 ,2). Therefore, the time points of the original audio are:
        conv1: (0+9)/2, (5+14)/2, ... = 4.5, 9.5, ... stride=5, kernel=10
        conv2: (4.5+14.5)/2, (14.5+24.5)/2, ... = 9.5, 19.5, ... stride=2, kernel=3
        conv3: (9.5+29.5)/2, (19.5+39.5)/2, ... = 19.5, 39.5, ... stride=2, kernel=3
        conv4: (19.5+59.5)/2, (39.5+79.5)/2, ... = 39.5, 79.5, ... stride=2, kernel=3
        conv5: (39.5+119.5)/2, (79.5+159.5)/2, ... = 79.5, 159.5, ... stride=2, kernel=3
        conv6: (79.5+159.5)/2, (239.5+319.5)/2, ... = 119.5, 279.5, ... stride=2, kernel=2
        conv7: (119.5+279.5)/2, (439.5+599.5)/2, ... = 199.5, 519.5, ... stride=2, kernel=2
    """
    HUBERT_model.eval()
    feature = "hubert"
    variants = [f"layer{i}" for i in range(19)]
    (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    ) = prepare_waveform(
        out_sr, wav_path, output_root, n_t, time_window, feature, variants
    )
    if not meta_only:
        waveform = torch.as_tensor(waveform)
        waveform = waveform.to(device)

        if device.type == "mps":
            half = False
        with autocast(device_type=device.type, enabled=half):
            HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE  #    HUBERT_bundle (HUBERTBundle): The HUBERT bundle containing the sample rate and other information.
            conv_kernels = [
                i[1] for i in HUBERT_bundle._params["extractor_conv_layer_config"]
            ]
            conv_strides = [
                i[2] for i in HUBERT_bundle._params["extractor_conv_layer_config"]
            ]

            if sample_rate != HUBERT_bundle.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, HUBERT_bundle.sample_rate
                )

            # pad 0.5s before and pad 0.5s after
            pad_number = int(HUBERT_bundle.sample_rate / 2)
            waveform = torch.nn.functional.pad(
                waveform, (pad_number, pad_number), "constant", 0
            )
            waveform = waveform.reshape(1, -1)
            hooks, activations = register_activation_hooks(HUBERT_model)
            with torch.inference_mode():
                features, _ = HUBERT_model.extract_features(waveform)
            remove_hooks(hooks)
            sr = HUBERT_bundle.sample_rate
            t_0 = -0.5

        for i, feats in enumerate(activations):
            # The start time of output of layer n is: t_0_{n+1} = t_0_{n} + (conv_kernel_{n}-1) / (2*sr_{n})
            # The sr of output of layer n is: sr_{n+1} = sr_{n} / conv_stride_{n}
            t_0 = t_0 + (conv_kernels[i] - 1) / (2 * sr)
            sr = sr / conv_strides[i]
            print(f"t_0_{i}={t_0}, sr_{i}={sr}")
            feature_variant_out_dir = feature_variant_out_dirs[i]
            feats = feats[0][0].squeeze(0)
            feats = feats.cpu().numpy().transpose()
            ### align time to start from 0 and make the length to be n_t
            t_length = feats.shape[0]
            t_origin = np.arange(t_length) / sr + t_0

            feats = align_time(feats, t_origin, t_new, "t f")
            print(f"Feature {i}: {feats.shape}")

            # save each layer as mat
            out_mat_path = os.path.join(
                feature_variant_out_dir, f"{wav_name_no_ext}.mat"
            )

            # save data as mat
            hdf5storage.savemat(
                out_mat_path, {"features": feats, "t": t_new + time_window[0]}
            )
            write_summary(
                feature_variant_out_dir,
                time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
                dimensions="[time, feature]",
                extra="Nothing",
            )
        for j, feats in enumerate(features):
            index = i + j + 1

            feature_variant_out_dir = feature_variant_out_dirs[index]
            feats = feats.squeeze(0)
            feats = feats.cpu().numpy()
            ### align time to start from 0 and make the length to be n_t
            t_length = feats.shape[0]
            t_origin = np.arange(t_length) / sr + t_0

            feats = align_time(feats, t_origin, t_new, "t f")
            print(f"Feature {index}: {feats.shape}")
            # save each layer as mat
            out_mat_path = os.path.join(
                feature_variant_out_dir, f"{wav_name_no_ext}.mat"
            )

            # save data as mat
            hdf5storage.savemat(
                out_mat_path, {"features": feats, "t": t_new + time_window[0]}
            )
            write_summary(
                feature_variant_out_dir,
                time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
                dimensions="[time, feature]",
                sampling_rate=out_sr,
                extra="Nothing",
            )
    else:
        for feature_variant_out_dir in feature_variant_out_dirs:
            write_summary(
                feature_variant_out_dir,
                time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
                dimensions="[time, feature]",
                sampling_rate=out_sr,
                extra="Nothing",
            )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project = "intracranial-natsound165"
    output_root = os.path.abspath(
        f"{__file__}/../../../projects_toy/{project}/analysis"
    )
    wav_dir = os.path.abspath(
        f"{__file__}/../../../projects_toy/intracranial-natsound165/stimuli/stimulus_audio"
    )
    stim_names = ["stim5_alarm_clock"]
    hubert(device, output_root, stim_names, wav_dir, out_sr=100, pc=100)
