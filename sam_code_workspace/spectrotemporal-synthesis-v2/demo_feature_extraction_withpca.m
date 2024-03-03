function spectrotemporal_features

% global root_directory;
% root_directory = my_root_directory;

% project_directory = [root_directory '/nima-lab-localizer-v2'];
% addpath([root_directory '/general-experiment-code']);
% addpath([root_directory '/general-audio-code']);
% addpath(genpath([root_directory '/ecog-analysis-code']));
root_directory = '/Users/snormanh/Desktop/projects';
addpath(genpath([root_directory '/spectrotemporal-synthesis-v2']));
addpath(genpath([root_directory '/general-analysis-code']));

% feature types
modulation_types = {'specmod', 'tempmod', 'spectempmod'};

% sampling rate to use for the analysis
sr = 50;

% number of PCs
nPCs = 100;

% stimuli
stimulus_directory = '/Users/snormanh/Desktop/projects/spectrotemporal-synthesis-v2/stimuli';
stim_names = {'man_speaking.wav', 'orchestra_music.wav', 'walking_on_leaves.wav'};
train_stim_names = {'orchestra_music.wav', 'walking_on_leaves.wav'};

% rates and scales
clear P;
% P.temp_mod_rates = [1 2 4 8 16 32];
% P.temp_mod_lowpass = [zeros(1,length(P.temp_mod_rates)-1)];
P.temp_mod_rates = [2 8];
P.temp_mod_lowpass = [0 0];
% P.spec_mod_rates = [0.25 0.5 1 2 4];
% P.spec_mod_lowpass = [zeros(1,length(P.spec_mod_rates)-1)];
P.spec_mod_rates = [0.5 2];
P.spec_mod_lowpass = [0 0];

% padding in time and frequency
% P.freq_pad_oct = 8;
% P.temp_pad_sec = 3;
P.freq_pad_oct = 1;
P.temp_pad_sec = 1;

% parameters of cochlear filtering
% P.audio_sr = 20000;
P.audio_sr = 4000;
P.lo_freq_hz = 100;
% P.lo_freq_hz = 50;
P.n_filts = round((freq2erb(P.audio_sr)-freq2erb(P.lo_freq_hz))/1.3581);
P.compression_factor = 0.3;
% P.overcomplete = 2;
P.overcomplete = 0;
% P.logf_spacing = 1/12;
P.logf_spacing = 1/6;
% P.env_sr = 400;
P.env_sr = 100;

% whether the filters should be causal
P.causal = false;

% nonlinearity to use
P.nonlin = 'modulus';

% stimuli to use to compute PC weights
P.stim_to_compute_PC_weights = train_stim_names;

% normalization
P.demean_feats = true;
P.std_feats = true;

% whether to explicitly compute the features and whether to overwrite
P.return_PCA_timecourses = false;
P.overwrite = false;

% directory to save features
feature_directory = '/Users/snormanh/Desktop/projects/spectrotemporal-synthesis-v2/demo-feature-extraction';

% pca timecourses
[pca_timecourse_MAT_files, pca_weight_MAT_files, model_features] = allfeats_pca_multistim(...
    modulation_types, stim_names, nPCs, sr, stimulus_directory, feature_directory, P);

% figure;
% hold on;
% n_model_features = length(model_features);
% h = nan(1, n_model_features);
% for i = 1:n_model_features
%     load(pca_weight_MAT_files{i}, 'pca_eigvals');
%     e = cumsum(pca_eigvals) / sum(pca_eigvals);
%     h(i) = plot(e, 'LineWidth', 2);
%     drawnow;
% end
% xlim([0 1001]);
% legend(h, model_features{:})
% load(pca_timecourse_MAT_files{1,4}, 'pca_timecourses');



