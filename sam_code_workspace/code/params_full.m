function P = params_full

% rates and scales
clear P;
P.temp_mod_rates = [4 1 2 4 8 16 32 64 128];
P.temp_mod_lowpass = [1 zeros(1,length(P.temp_mod_rates)-1)];
P.spec_mod_rates = [1 0.25 0.5 1 2 4];
P.spec_mod_lowpass = [1 zeros(1,length(P.spec_mod_rates)-1)];

% padding in time and frequency
% don't need temporal padding because that's being handled in the framing
P.freq_pad_oct = 8;
P.temp_pad_sec = 0;

% parameters of cochlear filtering
P.audio_sr = 20000;
P.lo_freq_hz = 50;
P.n_filts = round((freq2erb(P.audio_sr)-freq2erb(P.lo_freq_hz))/1.3581);
P.compression_factor = 0.3;
P.overcomplete = 2;
P.logf_spacing = 1/12;
P.env_sr = 400;