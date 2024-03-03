function combine_frames(framed_feature_directory, feature_name, orig_fnames, output_directory, feature_sr, target_dur, onset_in_frame)

for i = 1:length(orig_fnames)
	fname = orig_fnames{i};

	F = [];
	for j = 1:1e12

		% read in PC timecourses
		MAT_file = [framed_feature_directory '/' feature_name '_' fname '_frame' num2str(j) '.mat'];
		if ~exist(MAT_file, 'file')
			break;
		end
		X = load(MAT_file, 'F');
		F_stim = X.F;

		% select the target samples
		xi = (1:feature_sr*target_dur) + onset_in_frame*feature_sr;
		F = cat(1, F, F_stim(xi,:));

	end
	save(mkpdir([output_directory '/' fname '.mat']), 'F');
end
