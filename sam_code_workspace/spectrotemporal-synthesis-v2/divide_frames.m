function divide_frames(input_directory, fnames, output_directory, varargin)

% Divides long duration stimuli (e.g., > 10 seconds) into frames for computing
% spectrotemporal features.

clear I;
I.sr = 20e3;
I.target_dur = 10;
I.front_buffer_dur = 2; % amount to pad front to avoid wrap around
I.back_buffer_dur = 0.5; % amount to pad back to avoid wrap around
I = parse_optInputs_keyvalue(varargin, I);

frame_dur = I.target_dur + I.front_buffer_dur + I.back_buffer_dur;

for i = 1:length(fnames)

	[wav, wavsr] = audioread([input_directory '/' fnames{i} '.wav']);
	wav = wav(:,1);
	wav = resample(wav, I.sr, wavsr);
	waveform_dur = length(wav)/I.sr;
	n_frames = ceil(waveform_dur / I.target_dur);

	for j = 1:n_frames

		% all of the desired samples including invalid ones before start and after end of stim
		first_frame_smps = round(1-I.front_buffer_dur*I.sr):round(I.target_dur*I.sr + I.back_buffer_dur*I.sr);
		frame_smps = first_frame_smps + round((j-1)*I.target_dur*I.sr);

		% get the valid samples
		xi = frame_smps >= 1 & frame_smps <= length(wav);
		frame_wav = wav(frame_smps(xi));
		clear xi;

		% pad the begining and end to account for the invalid samples
		n_front_pad_smps = sum(frame_smps<1);
		n_back_pad_smps = sum(frame_smps>length(wav)); 
		if n_front_pad_smps > 0
			frame_wav = [zeros(n_front_pad_smps, 1); frame_wav];
		end
		if n_back_pad_smps > 0
			frame_wav = [frame_wav; zeros(n_back_pad_smps, 1)];
		end

		% save waveform and MAT files with target times in frames
		output_fname = [output_directory '/wav/' fnames{i} '_frame' num2str(j) '.wav'];
		audiowrite(mkpdir(output_fname), frame_wav, I.sr);

		MAT_file = [output_directory '/mat/' fnames{i} '_frame' num2str(j) '.mat'];
		target_time_in_frame = [0, I.target_dur] + I.front_buffer_dur;
		target_time_in_orig = [0, I.target_dur] + (j-1)*I.target_dur;
		save(mkpdir(MAT_file), 'target_time_in_frame', 'target_time_in_orig');

	end
end




