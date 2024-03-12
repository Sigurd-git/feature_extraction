import matlab.engine
import os
import numpy as np
import hdf5storage
import shutil
from utils.shared import write_summary, create_feature_variant_dir
from utils.pc import (
    generate_pca_pipeline,
    apply_pca_pipeline,
    generate_pca_pipeline_from_weights,
)
import torch


def generate_cochleagram_and_spectrotemporal(
    device,
    stim_names,
    wav_dir,
    out_sr,
    modulation_type,
    nonlin,
    output_root,
    debug=False,
    n_t=None,
    time_window=[-1, 1],
):
    """
    This function generates a cochleagram for a given audio file.

    Parameters:
    eng (object): An instance of the Matlab engine.
    wav_path (str): The path to the audio file for which the cochleagram is to be generated.
    output_root (str): The root directory where the cochleagram and its metadata will be stored.

    Returns:
    None. The function saves the cochleagram and its metadata to the output directory.
    """
    if debug:
        eng = matlab.engine.start_matlab("-desktop -r 'format short'")
        eng.open(
            os.path.abspath(
                f"{__file__}/../../../sam_code_workspace/code/coch_spectrotemporal.m"
            )
        )
    else:
        eng = matlab.engine.start_matlab()
    eng.addpath(
        eng.genpath(os.path.abspath(f"{__file__}/../../../sam_code_workspace/code"))
    )
    out_sr = float(out_sr)
    time_window = matlab.double(time_window)
    (coch_output_directory, modulation_output_directory, temp_dir, P) = (
        eng.coch_spectrotemporal(
            device,
            stim_names,
            wav_dir,
            out_sr,
            [modulation_type],
            nonlin,
            time_window,
            nargout=4,
        )
    )

    # create feature variant directories
    _, feature_original_cochleagram_dir = create_feature_variant_dir(
        output_root, "cochleagram", "original"
    )

    _, feature_variant_spectrotemporal_dir = create_feature_variant_dir(
        output_root, "spectrotemporal", f"{modulation_type}_{nonlin}"
    )
    time_window = np.array(time_window).reshape(-1)
    for stim_index, stim_name in enumerate(stim_names):
        # save cochleagram original
        coch_out_mat_path = os.path.join(
            feature_original_cochleagram_dir, f"{stim_name}.mat"
        )
        cochleagram_path = os.path.join(coch_output_directory, f"coch_{stim_name}.mat")
        cochleagrams = hdf5storage.loadmat(cochleagram_path)["F"]  # t x pc

        t_new = np.arange(cochleagrams.shape[0]) / out_sr + time_window[0]
        hdf5storage.savemat(coch_out_mat_path, {"features": cochleagrams, "t": t_new})

        # save spectrotemporal original
        spectrotemporal_out_mat_path = os.path.join(
            feature_variant_spectrotemporal_dir, f"{stim_name}.mat"
        )
        spectrotemporal_path = os.path.join(
            modulation_output_directory, f"{modulation_type}_{nonlin}_{stim_name}.mat"
        )
        spectrotemporals = hdf5storage.loadmat(spectrotemporal_path)[
            "F"
        ]  # time, frequency, [spectral modulation,] [temporal modulation,] rate
        # tempmod: 300,185,1,9
        # specmod: 300, 185, 7
        # spectempmod: 300, 185, 7, 9
        # spectempmod: 300, 185, 7, 9， 2
        # unsqueeze to make it consistent with the other two modulation types
        n_dims = len(spectrotemporals.shape)
        if n_dims == 4:
            spectrotemporals = np.expand_dims(spectrotemporals, axis=-1)
        if n_dims == 3:
            spectrotemporals = np.expand_dims(spectrotemporals, axis=-1)
            spectrotemporals = np.expand_dims(spectrotemporals, axis=-1)

        t_new = np.arange(spectrotemporals.shape[0]) / out_sr + time_window[0]
        hdf5storage.savemat(
            spectrotemporal_out_mat_path, {"features": spectrotemporals, "t": t_new}
        )
    P = hdf5storage.loadmat(cochleagram_path)["P"]
    f = P["f"][0, 0].reshape(-1)
    parameter_dict = {}
    for key in P.dtype.names:
        parameter_dict[key] = P[key][0, 0]
    parameter_dict["f"] = f
    # save parameters

    write_summary(
        feature_original_cochleagram_dir,
        time_window,
        "[time, feaquency]",
        extra="Parameters used to generate the cochleagram are contained in P variable of parameter.mat in the same directory.",
        parameter_dict=parameter_dict,
    )

    write_summary(
        feature_variant_spectrotemporal_dir,
        time_window,
        "[time, frequency, spectral modulation, temporal modulation, rate]",
        extra="Parameters used to generate the spectrotemporal are contained in P variable of parameter.mat in the same directory.",
        parameter_dict=parameter_dict,
    )
    # remove temp_dir
    shutil.rmtree(temp_dir)


def cochleagram_spectrotemporal(
    device,
    output_root,
    stim_names,
    wav_dir,
    out_sr=100,
    pc=100,
    time_window=[-1, 1],
    pca_weights_from=None,
    **kwargs,
):
    #     % % tempmod: only temporal modulation filters
    # % % specmod: only spectral modulation filters
    # % % spectempmod: joint spectrotemporal filters
    # % modulation_types = {'tempmod', 'specmod', 'spectempmod'};
    # nonlin: modulus or real or rect
    modulation_type = kwargs.get("modulation_type", "tempmod")
    nonlin = kwargs.get("nonlin", "modulus")
    debug = kwargs.get("debug", False)
    if isinstance(device, str):
        pass
    elif isinstance(device, torch.device):
        device = device.type
        if device == "cuda":
            device = "gpu"
        else:
            device = "cpu"
    generate_cochleagram_and_spectrotemporal(
        device=device,
        stim_names=stim_names,
        wav_dir=wav_dir,
        out_sr=out_sr,
        modulation_type=modulation_type,
        nonlin=nonlin,
        output_root=output_root,
        debug=debug,
        time_window=time_window,
    )
    feature_name = "cochleagram"
    variant = "original"
    if pc is not None:
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            feature_path = (
                f"{output_root}/features/{feature_name}/{variant}/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]
            wav_features.append(feature)
        print(f"Start computing PCs for {feature_name} ")
        if pca_weights_from is not None:
            weights_path = f"{pca_weights_from}/features/{feature_name}/{variant}/metadata/pca_weights.mat"
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
                variant=variant,
            )
        feature_variant_out_dir = apply_pca_pipeline(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            pca_pipeline=pca_pipeline,
            time_window=time_window,
        )
    feature_name = "spectrotemporal"
    variant = f"{modulation_type}_{nonlin}"
    if pc is not None:
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            feature_path = (
                f"{output_root}/features/{feature_name}/{variant}/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]
            feature = feature.reshape(feature.shape[0], -1)
            wav_features.append(feature)
        print(f"Start computing PCs for {feature_name} ")
        if pca_weights_from is not None:
            weights_path = f"{pca_weights_from}/features/{feature_name}/{variant}/metadata/pca_weights.mat"
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
                variant=variant,
            )
        feature_variant_out_dir = apply_pca_pipeline(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            pca_pipeline=pca_pipeline,
            time_window=time_window,
        )


if __name__ == "__main__":
    cochleagram_spectrotemporal(
        device=torch.device("cuda"),
        output_root=os.path.abspath(
            f"{__file__}/../../../projects_toy/intracranial-natsound165/analysis"
        ),
        stim_names=["stim5_alarm_clock", "stim7_applause"],
        wav_dir=os.path.abspath(
            f"{__file__}/../../../projects_toy/intracranial-natsound165/stimuli/audio"
        ),
        modulation_type="tempmod",
        debug=True,
    )
