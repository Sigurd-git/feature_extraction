
addpath(genpath('/scratch/snormanh_lab/shared/code/lab-analysis-code'));
addpath(genpath('/scratch/snormanh_lab/shared/code/spectrotemporal-synthesis-v2-master'));

% feature types
modulation_types = {'spectempmod'};

% sampling rate to use for the analysis
% 2 Hz is reasonable for fMRI
% 100 Hz is reasonable for intracranial data
feature_sr = 100;

% number of PCs
nPCs = 100;

% Parameters
P = measurement_parameters_default; 
P.env_sr = 100;
P.overcomplete = 1;


% whether the filters should be causal
P.causal = true;

% nonlinearity to use
% modulus means we will use an energy operator
P.nonlin = 'modulus';
% P.nonlin = 'real';

% normalization
P.demean_feats = true;
P.std_feats = false;

% whether to explicitly compute the features and whether to overwrite
P.return_PCA_timecourses = false;
P.overwrite = false;
P.save_highres = false;


% directory with the original stories and corresponding file names
input_directory = '/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v1/stimuli/stimulus_audio';
project_directory = '/home/gliao2/snormanh_lab_shared/projects/syllable-invariances_v1';
fnames = mydir(input_directory, '.wav');

% fnames = {'bigbang_audio', 'seinfeld_audio'};

% compute features and get PCs from frames
framed_feature_directory = [project_directory '/analysis/acoustics/modulus'];
[pca_timecourse_MAT_files, pca_weight_MAT_files, model_features, ...
    pca_timecourses_allstim_allmodels, pca_weights, pca_eigvals] = allfeats_pca_multistim(...
    modulation_types, fnames, nPCs, feature_sr, input_directory, framed_feature_directory, P);



