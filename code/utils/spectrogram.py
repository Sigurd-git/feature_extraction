import librosa
import os
import hdf5storage
import importlib
import numpy as np
from utils.pc import generate_pc

# from audio_tools import get_mel_spectrogram using importlib
import importlib.util
import sys

# Define the path to the module
module_path = "/home/gliao2/snormanh_lab_shared/code/audio_tools/audio_tools.py"

# Add the directory containing the module to sys.path
module_dir = os.path.dirname(module_path)
if module_dir not in sys.path:
    sys.path.append(module_dir)

# Load the module
spec = importlib.util.spec_from_file_location("audio_tools", module_path)
audio_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_tools)


# stim_names, output_root, wav_dir, out_sr, pc
def generate_spectrogram_features(out_sr, nfilts, wav_path, output_root, n_t=None):
    print(f"Getting mel spectrogram for stim: {wav_path}")
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "spectrogram")
    meta_out_path = os.path.join(output_root, "feature_metadata", "spectrogram.txt")
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(os.path.dirname(meta_out_path)):
        os.makedirs(os.path.dirname(meta_out_path))
    audio_waveform, aud_fs = librosa.load(wav_path, sr=None)
    t_num_new = int(np.round(audio_waveform.shape[0] / aud_fs * out_sr))
    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr
    t_num_new = len(t_new)
    # pad audio_waveform about 10/100 seconds
    audio_waveform = np.pad(audio_waveform, (0, int(10 / out_sr * aud_fs)), "constant")
    mel_spectrogram, freqs = audio_tools.get_mel_spectrogram(
        audio_waveform, aud_fs, steptime=1 / out_sr, nfilts=nfilts
    )
    mel_spectrogram = mel_spectrogram[:, :t_num_new].T
    feature_variant_out_dir = os.path.join(feature_class_out_dir, "original")

    if not os.path.exists(feature_variant_out_dir):
        os.makedirs(feature_variant_out_dir)
    # save each layer as mat
    out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")

    # save data as mat
    hdf5storage.savemat(out_mat_path, {"features": mel_spectrogram})
    # generate meta file for this layer
    with open(
        os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.txt"), "w"
    ) as f:
        f.write(f"""The shape of liberty spectrogram is {mel_spectrogram.shape}.""")
    pass


def spectrogram(device, output_root, stim_names, wav_dir, out_sr=100, pc=100, **kwargs):
    nfilts = kwargs.get("nfilts", 80)
    for stim_index, stim_name in enumerate(stim_names):
        wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
        generate_spectrogram_features(out_sr, nfilts, wav_path, output_root)

    if pc < nfilts:
        feature_name = "spectrogram"
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            feature_path = (
                f"{output_root}/features/{feature_name}/original/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]
            wav_features.append(feature)
        pca_pipeline = generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            demean=True,
            std=False,
        )
        generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            demean=True,
            std=False,
            pca_pipeline=pca_pipeline,
        )


if __name__ == "__main__":
    sr = 100
    nfilts = 80  # How many mel bands to return
    project = "syllable-invariances_v2"
    mat = hdf5storage.loadmat(
        f"/home/gliao2/snormanh_lab_shared/projects/{project}/stimuli/stim_names.mat"
    )
    stim_names = np.array([cell[0] for cell in mat["stim_names"].reshape(-1)])
    # remove random-durmatched-order1 from stim_names
    index = np.where(stim_names == "random-durmatched-order1")[0][0]
    stim_names = np.delete(stim_names, index)
    output_root = f"/scratch/snormanh_lab/shared/projects/{project}/analysis"
    wav_dir = f"/scratch/snormanh_lab/shared/projects/{project}/stimuli/stimulus_audio"
