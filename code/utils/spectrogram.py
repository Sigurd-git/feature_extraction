import os
import hdf5storage
import importlib
import numpy as np
from utils.pc import (
    generate_pca_pipeline,
    apply_pca_pipeline,
    generate_pca_pipeline_from_weights,
)
from utils.shared import write_summary, prepare_waveform

# from audio_tools import get_mel_spectrogram using importlib
import importlib.util
import sys

# Define the path to the module
module_path = os.path.abspath(f"{__file__}/../../../audio_tools/audio_tools.py")

# Add the directory containing the module to sys.path
module_dir = os.path.dirname(module_path)
if module_dir not in sys.path:
    sys.path.append(module_dir)

# Load the module
spec = importlib.util.spec_from_file_location("audio_tools", module_path)
audio_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_tools)


def spectrogram(
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
    nfilts = kwargs.get("nfilts", 80)
    if compute_original:
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            generate_spectrogram_features(
                out_sr,
                nfilts,
                wav_path,
                output_root,
                time_window=time_window,
                meta_only=meta_only,
            )

    if pc < nfilts:
        feature_name = "spectrogram"
        variant = "original"
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            feature_path = (
                f"{output_root}/features/{feature_name}/original/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]
            wav_features.append(feature)

        if pca_weights_from is not None:
            weights_path = f"{pca_weights_from}/features/{feature_name}/variant/metadata/pca_weights.mat"
            pca_pipeline = generate_pca_pipeline_from_weights(
                weights_from=weights_path, pc=pc
            )
        else:
            pca_pipeline = generate_pca_pipeline(
                wav_features,
                pc,
                output_root,
                feature_name,
                whiten=False,
                variant=variant,
            )
        apply_pca_pipeline(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            pca_pipeline=pca_pipeline,
            variant=variant,
            time_window=time_window,
            sampling_rate=out_sr,
            meta_only=meta_only,
        )


# stim_names, output_root, wav_dir, out_sr, pc
def generate_spectrogram_features(
    out_sr,
    nfilts,
    wav_path,
    output_root,
    n_t=None,
    time_window=[-1, 1],
    meta_only=False,
):
    feature = "spectrogram"
    variant = "original"
    if not meta_only:
        (
            wav_name_no_ext,
            waveform,
            sample_rate,
            t_num_new,
            t_new,
            feature_variant_out_dirs,
        ) = prepare_waveform(
            out_sr, wav_path, output_root, n_t, time_window, feature, [variant]
        )
        feature_variant_out_dir = feature_variant_out_dirs[0]
        # pad waveform about 10/100 seconds
        waveform = np.pad(waveform, (0, int(10 / out_sr * sample_rate)), "constant")
        mel_spectrogram, freqs = audio_tools.get_mel_spectrogram(
            waveform, sample_rate, steptime=1 / out_sr, nfilts=nfilts
        )
        mel_spectrogram = mel_spectrogram[:, :t_num_new].T

        # save each layer as mat
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")

        hdf5storage.savemat(
            out_mat_path, {"features": mel_spectrogram, "t": t_new + time_window[0]}
        )
    else:
        feature_variant_out_dir = os.path.join(
            output_root, "features", feature, variant
        )
    # generate meta file for this layer
    write_summary(
        feature_variant_out_dir,
        time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
        dimensions="[time, feature]",
        sampling_rate=out_sr,
        extra="Nothing",
    )


if __name__ == "__main__":
    sr = 100
    nfilts = 80  # How many mel bands to return
    project = "intracranial-natsound165"
    wav_path = os.path.abspath(
        f"{__file__}/../../../projects_toy/intracranial-natsound165/stimuli/stimulus_audio/stim5_alarm_clock.wav"
    )
    output_root = os.path.abspath(
        f"{__file__}/../../../projects_toy/{project}/analysis"
    )
    generate_spectrogram_features(
        sr, nfilts, wav_path, output_root, n_t=None, time_window=[-1, 1]
    )
