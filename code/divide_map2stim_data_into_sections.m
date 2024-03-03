global root_directory;
root_directory = '/home/gliao2/snormanh_lab_shared/projects/speech-long-TCI/data/UR15';
addpath(genpath(['/scratch/snormanh_lab/shared/code/lab-intracranial-code']));
addpath(genpath(['/scratch/snormanh_lab/shared/code/TCI-master']));
addpath(genpath(['/scratch/snormanh_lab/shared/code/general-audio-code']));
addpath(genpath(['/scratch/snormanh_lab/shared/code/general-analysis-code']));

exp = 'speech-long-TCI';
project_directory = ['/home/gliao2/snormanh_lab_shared/projects/speech-long-TCI-section/data'];

% subjids = {'UR1', 'UR4', 'UR5', 'UR6', 'HBRL625', 'HBRL634', 'UR11'};
subjids = {'UR15'};

%% Compute stimulus section onsets (from experimental stimuli)

N = 45; % # segments
n_groups = 4; % # scrambling groups
group_divisions = round(linspace(0, N, n_groups+1));
group_idx = group_divisions(1:end-1)+1;

stim_dur = 180; % length of total stimulus
seg_dur = 4; % length of each segment
section_border_sec = [0 cumsum(diff(group_divisions(1:end-1))*seg_dur) stim_dur];


%% Load neural data
% resp_win = [-1 181];
stim_to_use = 1:4; % story1 = 1:4, story2 = 5:8
mat_path = '/home/gliao2/snormanh_lab_shared/projects/speech-long-TCI/data/UR15/r1.mat'
for s = 1:length(subjids)
    [d, t, stim_names, electrode_research_numbers] = ...
        my_load_envelopes(exp, subjids{s}, mat_path, 'runs', 1); 
    start_idx = find(t==0);
    end_idx = find(t==stim_dur);
    data{s} = d(start_idx:end_idx,stim_to_use,:,:);
end

%% Divide neural data into sections
% NOTE TO SIGURD: This "D" structure is confusing and poorly organized, so
% you can change it if you'd like. But this is how I originally sent the
% data to you for the first several subjects.

env_sr = 100;
section_border_smp = (section_border_sec * env_sr) + 1;

% D dimensions -> time x rep x electrode
for i = 1:n_groups
    for s = 1:length(subjids)
        D = squeeze(data{s}(section_border_smp(i):section_border_smp(i+1)-1,3,:,:));
        save(mkpdir([project_directory '/UR15_fast_intact_section' num2str(i) '.mat']),'D','-v7.3')
        D = squeeze(data{s}(section_border_smp(i):section_border_smp(i+1)-1,1,:,:));
        save(mkpdir([project_directory '/UR15_fast_scram_section' num2str(i) '.mat']),'D','-v7.3')
        D = squeeze(data{s}(section_border_smp(i):section_border_smp(i+1)-1,4,:,:));
        save(mkpdir([project_directory '/UR15_slow_intact_section' num2str(i) '.mat']),'D','-v7.3')
        D = squeeze(data{s}(section_border_smp(i):section_border_smp(i+1)-1,2,:,:));
        save(mkpdir([project_directory '/UR15_slow_scram_section' num2str(i) '.mat']),'D','-v7.3')
    end
end



