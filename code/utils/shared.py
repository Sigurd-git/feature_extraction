import os
import torchaudio
import numpy as np
import hdf5storage
import matlab


def write_summary(
    feature_variant_out_dir,
    time_window="1 second before to 1 second after",
    dimensions="[time, feature]",
    sampling_rate=100,
    extra="Nothing",
    url="https://github.com/Sigurd-git/feature_extraction/tree/main/code/utils",
    parameter_dict=None,
):
    meta_out_dir = os.path.join(feature_variant_out_dir, "metadata")
    os.makedirs(meta_out_dir, exist_ok=True)

    meta_out_path = os.path.join(meta_out_dir, "summary.txt")
    # generate meta file for this feature
    with open(meta_out_path, "w") as f:
        f.write(
            f"""Time window: {time_window};
The dimensions of the matrices: {dimensions};
The sampling rate of the features: {sampling_rate} Hz;
Code used to generate those features are here: {url};
Extra comments: {extra}."""
        )
    if parameter_dict is not None:
        parameter_out_path = os.path.join(meta_out_dir, "parameters.mat")
        for key, value in parameter_dict.items():
            if isinstance(value, matlab.double):
                print(f"Converting {key} to numpy array")
                parameter_dict[key] = np.array(value._data)
        hdf5storage.savemat(parameter_out_path, parameter_dict)


def prepare_waveform(
    out_sr, wav_path, output_root, n_t, time_window, feature, variants
):
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    print(f"Getting {feature} features for stim: {wav_path}")
    feature_variant_out_dirs = [
        create_feature_variant_dir(output_root, feature, variant)[1]
        for variant in variants
    ]

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.reshape(-1)
    # pad waveform according to time_window
    assert time_window[0] <= 0 and time_window[1] >= 0
    before_pad = int(abs(time_window[0]) * sample_rate)
    after_pad = int(abs(time_window[1]) * sample_rate)
    waveform = np.pad(waveform, (before_pad, after_pad), "constant")

    t_num_new = int(np.round(waveform.shape[0] / sample_rate * out_sr))
    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr
    t_num_new = len(t_new)

    return (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    )


def create_feature_variant_dir(output_root, feature, variant):
    feature_class_out_dir = os.path.join(output_root, "features", feature)
    feature_variant_out_dir = os.path.join(feature_class_out_dir, variant)
    os.makedirs(feature_variant_out_dir, exist_ok=True)
    return feature_class_out_dir, feature_variant_out_dir
