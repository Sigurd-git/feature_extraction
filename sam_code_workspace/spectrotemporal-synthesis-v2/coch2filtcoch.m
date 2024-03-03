function [filtcoch, Hts] = coch2filtcoch(coch, spec_mod_rate, temp_mod_rate, P, ...
    spec_mod_lowpass, temp_mod_lowpass, complex_filters, separable, causal, ...
    fourier_domain, spec_BW, temp_BW, spec_wavelet, temp_wavelet, ...
    spec_random_phase, temp_random_phase, spec_random_filt, temp_random_filt, ...
    random_seed, delay)

if nargin < 5 || isempty(spec_mod_lowpass)
    spec_mod_lowpass = false;
end

if nargin < 6 || isempty(temp_mod_lowpass)
    temp_mod_lowpass = false;
end

if nargin < 7 || isempty(complex_filters)
    complex_filters = false;
end

if nargin < 8 || isempty(separable)
    separable = true;
end

if nargin < 9 || isempty(causal)
    causal = true;
end

if nargin < 10 || isempty(fourier_domain)
    fourier_domain = false;
end

if nargin < 11 || isempty(spec_BW)
    spec_BW = 1;
end

if nargin < 12 || isempty(temp_BW)
    temp_BW = 1;
end

if nargin < 13
    spec_wavelet = 'mexicanhat';
end

if nargin < 14 || isempty(temp_wavelet)
    temp_wavelet = 'gammatone';
end

if nargin < 15 || isempty(spec_random_phase)
    spec_random_phase = false;
end

if nargin < 16 || isempty(temp_random_phase)
    temp_random_phase = false;
end

if nargin < 17 || isempty(spec_random_filt)
    spec_random_filt = false;
end

if nargin < 18 || isempty(temp_random_filt)
    temp_random_filt = false;
end

if nargin < 19 || isempty(random_seed)
    random_seed = false;
end

if nargin < 20 || isempty(delay)
    delay = 0;
end


% FT of the cochleogram
FT_coch = fft2(coch);

% impulse response
Hts = filt_spectemp_mod(...
    spec_mod_rate, temp_mod_rate, ...
    size(coch,2), size(coch,1), P, spec_mod_lowpass, ...
    temp_mod_lowpass, 0, 0, complex_filters, separable, causal, ...
    spec_BW, temp_BW, spec_wavelet, temp_wavelet, ...
    spec_random_phase, temp_random_phase, ...
    spec_random_filt, temp_random_filt, random_seed, delay);

% convolve
if fourier_domain
    filtcoch = FT_coch .* Hts;
else
    filtcoch = ifft2(FT_coch .* Hts);
end

% ensure real (only needed because of numerical issues)
try
    if ~complex_filters && ~fourier_domain
        filtcoch = real(filtcoch);
    end
catch
    keyboard
end