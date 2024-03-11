import matlab.engine
import os
import numpy as np
import hdf5storage
import shutil
from utils.shared import write_summary, create_feature_variant_dir
import torch


def generate_cochleagram_and_spectrotemporal(
    device,
    stim_names,
    wav_dir,
    out_sr,
    pc,
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
    (
        pca_weight_MAT_files,
        model_features,
        pca_timecourses_allstim_allmodels,
        coch_output_directory,
        modulation_output_directory,
        pca_output_directories,
        temp_dir,
        P,
    ) = eng.coch_spectrotemporal(
        device,
        stim_names,
        wav_dir,
        out_sr,
        pc,
        [modulation_type],
        nonlin,
        time_window,
        nargout=8,
    )
    coch_pca_directory = os.path.abspath(pca_output_directories[0])
    spectrotemporal_pca_directory = os.path.abspath(pca_output_directories[1])
    coch_pca_weight_MAT_file = os.path.abspath(pca_weight_MAT_files[0]) + ".mat"
    spectrotemporal_pca_weight_MAT_file = (
        os.path.abspath(pca_weight_MAT_files[1]) + ".mat"
    )

    # create feature variant directories
    _, feature_original_cochleagram_dir = create_feature_variant_dir(
        output_root, "cochleagram", "original"
    )

    _, feature_pc_cochleagram_dir = create_feature_variant_dir(
        output_root, "cochleagram", f"pc{pc}"
    )

    _, feature_pc_spectrotemporal_dir = create_feature_variant_dir(
        output_root, "spectrotemporal", f"{modulation_type}_{nonlin}_pc{pc}"
    )

    _, feature_variant_spectrotemporal_dir = create_feature_variant_dir(
        output_root, "spectrotemporal", f"{modulation_type}_{nonlin}"
    )
    time_window = np.array(time_window).reshape(-1)
    for stim_index, stim_name in enumerate(stim_names):
        # save cochleagram pc
        coch_out_mat_path = os.path.join(feature_pc_cochleagram_dir, f"{stim_name}.mat")
        cochleagram_path = os.path.join(coch_pca_directory, f"coch_{stim_name}.mat")
        cochleagrams = hdf5storage.loadmat(cochleagram_path)["F"]  # t x pc
        P = hdf5storage.loadmat(cochleagram_path)["P"]
        t_new = np.arange(cochleagrams.shape[0]) / out_sr + time_window[0]
        hdf5storage.savemat(coch_out_mat_path, {"features": cochleagrams, "t": t_new})

        # save cochleagram original
        coch_out_mat_path = os.path.join(
            feature_original_cochleagram_dir, f"{stim_name}.mat"
        )
        cochleagram_path = os.path.join(coch_output_directory, f"coch_{stim_name}.mat")
        cochleagrams = hdf5storage.loadmat(cochleagram_path)["F"]  # t x pc
        t_new = np.arange(cochleagrams.shape[0]) / out_sr + time_window[0]
        hdf5storage.savemat(coch_out_mat_path, {"features": cochleagrams, "t": t_new})

        # save spectrotemporal pc
        spectrotemporal_pc_out_mat_path = os.path.join(
            feature_pc_spectrotemporal_dir, f"{stim_name}.mat"
        )
        spectrotemporal_path = os.path.join(
            spectrotemporal_pca_directory, f"{modulation_type}_{nonlin}_{stim_name}.mat"
        )
        spectrotemporals = hdf5storage.loadmat(spectrotemporal_path)["F"]  # t x pc
        t_new = np.arange(spectrotemporals.shape[0]) / out_sr + time_window[0]
        V = hdf5storage.loadmat(spectrotemporal_pca_weight_MAT_file)["pca_weights"]
        hdf5storage.savemat(
            spectrotemporal_pc_out_mat_path,
            {"features": spectrotemporals, "t": t_new},
        )

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


    pca_weights_out_mat_path_coch_original = os.path.join(
        feature_original_cochleagram_dir, "pca_weights.mat"
    )
    V = hdf5storage.loadmat(coch_pca_weight_MAT_file)["pca_weights"]
    hdf5storage.savemat(pca_weights_out_mat_path_coch_original, {"V": V})

    pca_weights_out_mat_path_coch_pc = os.path.join(
        feature_pc_cochleagram_dir, "pca_weights.mat"
    )
    hdf5storage.savemat(pca_weights_out_mat_path_coch_pc, {"V": V})

    pca_weights_out_mat_path_spectrotemporal_pc = os.path.join(
        feature_pc_spectrotemporal_dir, "pca_weights.mat"
    )
    V = hdf5storage.loadmat(spectrotemporal_pca_weight_MAT_file)["pca_weights"]
    hdf5storage.savemat(pca_weights_out_mat_path_spectrotemporal_pc, {"V": V})
    pca_weights_out_mat_path_spectrotemporal = os.path.join(
        feature_variant_spectrotemporal_dir, "pca_weights.mat"
    )
    hdf5storage.savemat(pca_weights_out_mat_path_spectrotemporal, {"V": V})

    write_summary(
        feature_original_cochleagram_dir,
        time_window,
        "[time, feaquency]",
        extra="Parameters used to generate the cochleagram are contained in P variable of parameter.mat in the same directory.",
        parameter_dict=P,
    )
    write_summary(
        feature_pc_cochleagram_dir,
        time_window,
        "[time, pc]",
        extra="Parameters used to generate the cochleagram are contained in P variable of parameter.mat in the same directory. The pca_weight used to compute the pcs are saved together with the pcs in the same mat file. The pca_weight is saved in the V variable.",
        parameter_dict=P,
    )
    write_summary(
        feature_pc_spectrotemporal_dir,
        time_window,
        "[time, pc]",
        extra="Parameters used to generate the spectrotemporal are contained in P variable of parameter.mat in the same directory. The pca_weight used to compute the pcs are saved together with the pcs in the same mat file. The pca_weight is saved in the V variable.",
        parameter_dict=P,
    )
    write_summary(
        feature_variant_spectrotemporal_dir,
        time_window,
        "[time, frequency, spectral modulation, temporal modulation, rate]",
        extra="Parameters used to generate the spectrotemporal are contained in P variable of parameter.mat in the same directory.",
        parameter_dict=P,
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
    **kwargs,
):
    #     % % tempmod: only temporal modulation filters
    # % % specmod: only spectral modulation filters
    # % % spectempmod: joint spectrotemporal filters
    # % modulation_types = {'tempmod', 'specmod', 'spectempmod'};
    # nonlin: modulus or real or rect
    modulation_type = kwargs.get("modulation_type", ["tempmod"])
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
        pc=pc,
        modulation_type=modulation_type,
        nonlin=nonlin,
        output_root=output_root,
        debug=debug,
        time_window=time_window,
    )


if __name__ == "__main__":
    cochleagram_spectrotemporal(
        device=torch.device("gpu"),
        output_root=os.path.abspath(
            f"{__file__}/../../../projects_toy/intracranial-natsound165/analysis"
        ),
        stim_names=["stim5_alarm_clock", "stim7_applause"],
        wav_dir=os.path.abspath(
            f"{__file__}/../../../projects_toy/intracranial-natsound165/stimuli/stimulus_audio"
        ),
        modulation_type="tempmod",
        debug=True,
    )
