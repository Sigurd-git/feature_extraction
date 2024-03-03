% Simple code illustrating how to do analysis and synthesis using the filters
% from the spectrotemporal model.
% 
% 2017-05-10: Created, Sam NH

%% Setup

addpath('Sound_Texture_Synthesis_Toolbox');

% Parameters
P = measurement_parameters_default;
P.env_sr = 100;
P.overcomplete = 1;
P.logf_spacing = 1/12;

% read in waveform
[wav,sr] = audioread('speech.wav');
wav = resample(wav, P.audio_sr, sr);
wav = wav(1:P.audio_sr);

%% Cochleogram

% cochleogram
[coch, P, R, audio_filts, low_cutoffs] = wav2coch_without_filts(wav, P);

%% plot cochleogram
figure;
plot_cochleogram(coch, P.f, P.t);

% visualize the filters
figure;
plot(low_cutoffs, audio_filts(1:500:end,:)', 'k-');
xlabel('Frequency'); ylabel('Filter Amplitude');

%% Single subband

clc;
% close all;
addpath('2DFT');

% parameters
spec_mod_rate = 2; % cycles per octave
temp_mod_rate = 2; % Hertz
spec_mod_lowpass = false;
temp_mod_lowpass = false;
complex_filters = false;
causal = true;
fourier_domain = false;
separable = false;
spec_BW = 1;
temp_BW = 1;
spec_wavelet = 'mexicanhat';
temp_wavelet = 'gammatone';
spec_random_phase = false;
temp_random_phase = false;
spec_random_filt = false;
temp_random_filt = false;
random_seed = 4;

% result of convolution with the impulse response
coch_subband = coch2filtcoch(coch, spec_mod_rate, temp_mod_rate, P, ...
    spec_mod_lowpass, temp_mod_lowpass, complex_filters, separable, causal, ...
    fourier_domain, spec_BW, temp_BW, spec_wavelet, temp_wavelet, ...
    spec_random_phase, temp_random_phase, ...
    spec_random_filt, temp_random_filt, random_seed);

% plot subband
figure;
plot_cochleogram(real(coch_subband), P.f, P.t);

% impulse response in 2D fourier domain
Hts_FT = filt_spectemp_mod(...
    spec_mod_rate, temp_mod_rate, ...
    size(coch,2), size(coch,1), P, spec_mod_lowpass, ...
    temp_mod_lowpass, 0, 0, complex_filters, separable, causal, ...
    spec_BW, temp_BW, spec_wavelet, temp_wavelet, ...
    spec_random_phase, temp_random_phase, ...
    spec_random_filt, temp_random_filt, random_seed);

% impulse response in signal domain
Hts_signal = ifft2(Hts_FT)';
Hts_signal = flipud(circshift(Hts_signal, [ceil(size(coch,2)/2-1), 0]));

% plot impulse response
figure;
imagesc((real(Hts_signal)), ...
    max(abs(real(Hts_signal(:))))*[-1 1]);

%% Multiple subbands

% include negative and positive rates
% P.temp_mod_rates = unique([-P.temp_mod_rates, P.temp_mod_rates]);

% compute subbands
% time x frequency x spectral modulation x temporal modulation x orientation
P.temp_mod_rates = [1,2,4,8,16,32];
P.temp_mod_lowpass = zeros(size(P.temp_mod_rates));
P.spec_mod_rates = [0.25,0.5,1,2,4];
P.spec_mod_lowpass = zeros(size(P.spec_mod_rates));
complex = true;
filtcoch_allsubbands = coch2filtcoch_allsubbands(coch, P, complex);

% plot upwards and downwards
figure;
plot_cochleogram(real(filtcoch_allsubbands(:,:,2,3,1)), P.f, P.t);
title('Upwards')
figure
plot_cochleogram(real(filtcoch_allsubbands(:,:,2,3,2)), P.f, P.t);
title('Downwards')

% plot real and imaginary components
figure;
plot_cochleogram(real(filtcoch_allsubbands(:,:,2,3,1)), P.f, P.t);
title('Real')
figure
plot_cochleogram(imag(filtcoch_allsubbands(:,:,2,3,1)), P.f, P.t);
title('Imaginary')

% energy plot
figure;
plot_cochleogram(abs(filtcoch_allsubbands(:,:,2,3,1)), P.f, P.t);
title('energy')

