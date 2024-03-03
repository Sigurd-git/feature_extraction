
addpath(genpath('/scratch/snormanh_lab/shared/code/general-analysis-code'));
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
P.causal = false;

% nonlinearity to use
% modulus means we will use an energy operator
P.nonlin = 'modulus';

% normalization
P.demean_feats = true;
P.std_feats = false;

% whether to explicitly compute the features and whether to overwrite
P.return_PCA_timecourses = false;
P.overwrite = false;
P.save_highres = false;


input_directory_root = '/scratch/snormanh_lab/shared/sounds/slakh2100/slakh_60sec_excerpts_stems'
output_directory_root = '/scratch/snormanh_lab/shared/kcriss/slakh_features/slakh_60sec_excerpts_stems';

contents = mydir(input_directory_root);
for i = 1:length(contents)


        sub_dir = strcat(input_directory_root, '/', char(contents(i)));
        sub_contents = mydir(sub_dir);

        for j = 1:length(sub_contents)

            sub_sub_dir = strcat(sub_dir, '/', char(sub_contents(j)));

            % compute features and get PCs from frames
            input_directory = sub_sub_dir;
            output_directory = strcat(output_directory_root,'/',char(contents(i)),'/',char(sub_contents(j)));
            stimuli = mydir(input_directory);
            [pca_timecourse_MAT_files, pca_weight_MAT_files, model_features,pca_timecourses_allstim_allmodels, pca_weights, pca_eigvals] = allfeats_pca_multistim(modulation_types, stimuli, nPCs, feature_sr,input_directory, output_directory, P);


        end


end



input_directory_root = '/scratch/snormanh_lab/shared/sounds/slakh2100/slakh_60sec_excerpts_variation_v4'
output_directory_root = '/scratch/snormanh_lab/shared/kcriss/slakh_features/slakh_60sec_excerpts_variation_v4';
% compute features and get PCs from frames
input_directory = input_directory_root
output_directory = output_directory_root;
stimuli = mydir(input_directory);
[pca_timecourse_MAT_files, pca_weight_MAT_files, model_features,pca_timecourses_allstim_allmodels, pca_weights, pca_eigvals] = allfeats_pca_multistim(modulation_types, stimuli, nPCs, feature_sr,input_directory, output_directory, P);




input_directory_root = '/scratch/snormanh_lab/shared/sounds/slakh2100/slakh_60sec_excerpts_variation_v4_levelramp-4-1.5sec-15dB/wav'
output_directory_root = '/scratch/snormanh_lab/shared/kcriss/slakh_features/slakh_60sec_excerpts_variation_v4_levelramp-4-1.5sec-15dB/wav';

input_directory = input_directory_root
output_directory = output_directory_root;
stimuli = mydir(input_directory);
[pca_timecourse_MAT_files, pca_weight_MAT_files, model_features,pca_timecourses_allstim_allmodels, pca_weights, pca_eigvals] = allfeats_pca_multistim(modulation_types, stimuli, nPCs, feature_sr,input_directory, output_directory, P);


