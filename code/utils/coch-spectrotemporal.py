import matlab.engine
import os
import numpy as np
import hdf5storage


def generate_cochleagram_and_spectrotemporal(
    stim_names,
    wav_dir,
    out_sr,
    pc,
    modulation_type,
    nonlin,
    output_root,
    debug=False,
    n_t=None,
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

    (
        pca_weight_MAT_files,
        model_features,
        pca_timecourses_allstim_allmodels,
        coch_output_directory,
        modulation_output_directory,
        pca_output_directories,
    ) = eng.coch_spectrotemporal(
        stim_names, wav_dir, out_sr, pc, [modulation_type], nonlin, nargout=6
    )
    coch_pca_directory = pca_output_directories[0]
    spectrotemporal_pca_directory = pca_output_directories[1]

    # coch
    meta_out_dir = os.path.join(output_root, "feature_metadata")

    coch_class_out_dir = os.path.join(output_root, "features", "cochleagram")
    feature_variant_cochleagram_dir = (
        os.path.join(coch_class_out_dir, "original")
        if pc is None
        else os.path.join(coch_class_out_dir, f"pc{pc}")
    )
    spectrotemporal_class_out_dir = os.path.join(
        output_root, "features", "spectrotemporal"
    )
    feature_variant_spectrotemporal_dir = (
        os.path.join(spectrotemporal_class_out_dir, f"{modulation_type}_{nonlin}")
        if pc is None
        else os.path.join(
            spectrotemporal_class_out_dir, f"{modulation_type}_{nonlin}_pc{pc}"
        )
    )

    if not os.path.exists(feature_variant_cochleagram_dir):
        os.makedirs(feature_variant_cochleagram_dir)
    if not os.path.exists(feature_variant_spectrotemporal_dir):
        os.makedirs(feature_variant_spectrotemporal_dir)
    if not os.path.exists(meta_out_dir):
        os.makedirs(meta_out_dir)

    for stim_index, stim_name in enumerate(stim_names):
        coch_out_mat_path = os.path.join(
            feature_variant_cochleagram_dir, f"{stim_name}.mat"
        )
        cochleagram_path = os.path.join(coch_pca_directory, f"coch_{stim_name}.mat")
        cochleagrams = hdf5storage.loadmat(cochleagram_path)["F"]  # t x pc
        # clip or pad so that the number of time steps is n_t
        if n_t is not None:
            if cochleagrams.shape[0] > n_t:
                print(f"clip:{stim_name} from {cochleagrams.shape[0]} to {n_t}")
                cochleagrams = cochleagrams[:n_t, :]
            elif cochleagrams.shape[0] < n_t:
                print(f"pad:{stim_name} from {cochleagrams.shape[0]} to {n_t}")
                cochleagrams = np.pad(
                    cochleagrams, ((0, n_t - cochleagrams.shape[0]), (0, 0))
                )

        # save data as mat
        hdf5storage.savemat(coch_out_mat_path, {"features": cochleagrams})

        # generate meta file for cochleagram
        with open(
            os.path.join(feature_variant_cochleagram_dir, f"{stim_name}.txt"), "w"
        ) as f:
            f.write(f"The shape of this feature is {cochleagrams.shape}.")

        spectrotemporal_out_mat_path = os.path.join(
            feature_variant_spectrotemporal_dir, f"{stim_name}.mat"
        )
        spectrotemporal_path = os.path.join(
            spectrotemporal_pca_directory, f"{stim_name}.mat"
        )
        spectrotemporals = hdf5storage.loadmat(spectrotemporal_path)["F"]  # t x pc
        if n_t is not None:
            if spectrotemporals.shape[0] > n_t:
                print(f"clip:{stim_name} from {spectrotemporals.shape[0]} to {n_t}")
                spectrotemporals = spectrotemporals[:n_t, :]
            elif spectrotemporals.shape[0] < n_t:
                print(f"pad:{stim_name} from {spectrotemporals.shape[0]} to {n_t}")
                spectrotemporals = np.pad(
                    spectrotemporals, ((0, n_t - spectrotemporals.shape[0]), (0, 0))
                )

        hdf5storage.savemat(
            spectrotemporal_out_mat_path, {"features": spectrotemporals}
        )
        with open(
            os.path.join(feature_variant_spectrotemporal_dir, f"{stim_name}.txt"), "w"
        ) as f:
            f.write(
                f"The shape of this feature is {spectrotemporals.shape}."
            )  # (n_time,n_frequency)

    # generate meta file for cochleagram
    with open(os.path.join(meta_out_dir, "cochleagram_original.txt"), "w") as f:
        f.write(
            """Timestamps start from 0, sr=100Hz. You can find out the shape of each stimulus in features/cochleagram/original/stimuli.txt as (n_time,n_frequency). 
The timing (in seconds) of each time stamp can be computed like: timing=np.arange(n_time)/sr.
                
Parameters for computing the cochleagrams are:
% maximum duration of the input and synthesis sound in seconds
P.max_duration_sec = 12;

% center frequencies of the temporal modulation filters in Hz
% 0 indicates a filter with only power at the DC
% wether or not the rates are lowpass or bandpass (the default)
P.temp_mod_rates = [4,1,2,4,8,16,32,64,128];
P.temp_mod_lowpass = [1, zeros(1,8)];

% center frequencies of the spectral modulation filters in cyc/octave
% 0 indicates a filter with only power at the DC
% wether or not the scales are lowpass or bandpass (the default)
P.spec_mod_rates = [1,0.25,0.5,1,2,4,8];
P.spec_mod_lowpass = [1, zeros(1,6)];

% amount of frequency padding, twice the period of the lowest spectral scale
P.freq_pad_oct = 2/min(P.spec_mod_rates(P.spec_mod_rates>0));

% temporal padding
% 3x the longest period in the synthesis
all_temp_rates = P.temp_mod_rates(P.temp_mod_rates>0);
P.temp_pad_sec = 3/min(all_temp_rates);

% audio sampling rate
P.audio_sr = 20000;

% sampling rate of the envelope in seconds
P.env_sr = 100;

% lowest filter in the audio filter bank
% highest is the nyquist - P.audio_sr/2
P.lo_freq_hz = 50;

% number of cosine filters to use
% increasing the number of filters
% decreases the bandwidth of the filters
P.n_filts = round((freq2erb(P.audio_sr)-freq2erb(P.lo_freq_hz))/1.3581);

% whether or not the number of filters is
% complete (=0), 1x overcomplete (=1), or 2x overcomplete (=2)
% overcomplete representations typically result in slightly more compelling 
% synthetics, but require more time and memory 
P.overcomplete = 1;

% frequency spacing of samples after interpolation to a logarithmic scale
% in octaves
P.logf_spacing = 1/12;

% factor to which cochleogram envelopes are raised
P.compression_factor = 0.3;

% whether or note the filters are causal in time
P.causal = true;"""
        )


def cochleagram_and_spectrotemporal(
    device, output_root, stim_names, wav_dir, out_sr=100, pc=100, **kwargs
):
    #     % % tempmod: only temporal modulation filters
    # % % specmod: only spectral modulation filters
    # % % spectempmod: joint spectrotemporal filters
    # % modulation_types = {'tempmod', 'specmod', 'spectempmod'};
    # nonlin: modulus or real or rect
    modulation_type = kwargs.get("modulation_type", ["tempmod"])
    nonlin = kwargs.get("nonlin", "modulus")
    debug = kwargs.get("debug", False)
    generate_cochleagram_and_spectrotemporal(
        stim_names=stim_names,
        wav_dir=wav_dir,
        out_sr=out_sr,
        pc=pc,
        modulation_type=modulation_type,
        nonlin=nonlin,
        output_root=output_root,
        debug=debug,
    )


if __name__ == "__main__":
    cochleagram_and_spectrotemporal(
        device="cuda",
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
