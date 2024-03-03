function [fnames] = coch_spectrotemporal(stim_names, wav_dir, out_sr,pc,modulation_types,nonlin);
% function [pca_timecourse_MAT_files, pca_weight_MAT_files, model_features, pca_timecourses_allstim_allmodels, coch_output_directory, modulation_output_directory, pca_output_directories] = coch_spectrotemporal(input_directory, stim_names, wav_dir, out_sr,pc);

root_directory = '/scratch/snormanh_lab/shared/Sigurd/sam_code_workspace/projects';
project_directory = [root_directory '/intracranial-natsound165'];
addpath(genpath([root_directory '/spectrotemporal-synthesis-v2']));
addpath(genpath([root_directory '/general-analysis-code']));
addpath('/scratch/snormanh_lab/shared/Sigurd/sam_code_workspace/code');

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
P.overwrite = false;
P.save_highres = true;


% directory with the original stories and corresponding file names
fnames = strcat(stim_names, '.wav');
%% 

% % add silence to the end of the waveforms
% for i = 1:length(fnames)
%     [wav, sr] = audioread([wav_dir '/' fnames{i}]);
%     wav = [wav; zeros(sr,1)];
%     output_file = [wav_dir '-1sec-pad/' fnames{i}];
%     audiowrite(mkpdir(output_file), wav, sr);
% end
% %% 

% % compute features and get PCs
% feature_directory = [project_directory '/features/spectemp-with-padding'];
% [pca_timecourse_MAT_files, pca_weight_MAT_files, model_features, pca_timecourses_allstim_allmodels, coch_output_directory, modulation_output_directory, pca_output_directories] = allfeats_pca_multistim(...
%     modulation_types, fnames, nPCs, feature_sr, [input_directory '-1sec-pad'], feature_directory, P);


end