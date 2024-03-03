function F_struct = get_coch_features(wav_path);

addpath(genpath('/scratch/snormanh_lab/shared/code/general-analysis-code'));
addpath(genpath('/scratch/snormanh_lab/shared/code/spectrotemporal-synthesis-v2-master'));


% using very coarse parameters for speed
P = measurement_parameters_default; 
P.env_sr = 100;
P.overcomplete = 1;
P.logf_spacing = 1/12;



% read in stimulus
[y, wav_sr] = audioread(wav_path);

% convert to mono if necessary
if size(y,2) > 1
    y = mean(y,2);
end

% resample
if P.audio_sr ~= wav_sr
    y = resample(y, P.audio_sr, wav_sr);
end

% pad in time
wav_dur_sec = length(y) / P.audio_sr;
y = [y; zeros(P.audio_sr * P.temp_pad_sec,1)]; %#ok<AGROW>

% cochleogram
fprintf('Computing cochleagram\n'); drawnow;
[coch, P_coch] = wav2coch_without_filts(y, P);

% remove temporal padding
F = coch((1:wav_dur_sec*P_coch.env_sr), :);


F_struct = struct('coch', F, 'P', P_coch);

end

