function P = params_toy

% rates and scales
clear P;
P.temp_mod_rates = [1 4];
P.temp_mod_lowpass = [0 0];
P.spec_mod_rates = [0.5 2];
P.spec_mod_lowpass = [0 0];

% padding in time and frequency
% don't need temporal padding because that's being handled in the framing
P.freq_pad_oct = 1;
P.temp_pad_sec = 0;

% parameters of cochlear filtering
P.audio_sr = 4000;
P.lo_freq_hz = 100;
P.n_filts = round((freq2erb(P.audio_sr)-freq2erb(P.lo_freq_hz))/1.3581);
P.compression_factor = 0.3;
P.overcomplete = 0;
P.logf_spacing = 1/6;
P.env_sr = 100;