function [pca_weight_MAT_files, model_features, pca_timecourses_allstim_allmodels, coch_output_directory, modulation_output_directory, pca_output_directories,temp_dir,P] = coch_spectrotemporal(device,stim_names, wav_dir, out_sr, pc, modulation_types, nonlin,time_window);

fullPath = mfilename('fullpath');
[currentFolder, ~, ~] = fileparts(fullPath);

project_directory = [currentFolder '/../project'];
addpath(genpath([currentFolder '/../spectrotemporal-synthesis-v2']));
addpath(genpath([currentFolder '/../general-analysis-code']));
addpath(currentFolder);

% sampling rate to use for the analysis
feature_sr = out_sr;

% number of PCs
nPCs = pc;

% Parameters
P = measurement_parameters_default; 
P.env_sr = 100;
P.overcomplete = 1;


% whether the filters should be causal
P.causal = true;

% nonlinearity to use
% modulus means we will use an energy operator
P.nonlin = nonlin;
% P.nonlin = 'real';

% normalization
P.demean_feats = true;
P.std_feats = false;

% whether to explicitly compute the features and whether to overwrite
P.return_PCA_timecourses = false;
P.overwrite = true;
P.save_highres = true;
P.device = device;

% directory with the original stories and corresponding file names
fnames = strcat(stim_names, '.wav');

jobID = getenv('SLURM_JOB_ID');
%% 

temp_dir = [project_directory '/temp_'  jobID];
wav_temp_dir = [temp_dir '/wav'];
if ~exist(wav_temp_dir, 'dir')
    mkdir(wav_temp_dir);
end

before = abs(time_window(1));
after = abs(time_window(2));

% add silence to the end of the waveforms
for i = 1:length(fnames)
    [wav, sr] = audioread([wav_dir '/' fnames{i}]);
    wav = [zeros(before*sr,1); wav; zeros(after*sr,1)];
    output_file = [wav_temp_dir '/' fnames{i}];
    audiowrite(mkpdir(output_file), wav, sr);
end
%% 

% compute features and get PCs
feature_directory = [temp_dir '/spectemp-with-padding'];
[pca_timecourse_MAT_files, pca_weight_MAT_files, model_features, pca_timecourses_allstim_allmodels, coch_output_directory, modulation_output_directory, pca_output_directories] = allfeats_pca_multistim(...
    modulation_types, fnames, nPCs, feature_sr, wav_temp_dir, feature_directory, P);


end