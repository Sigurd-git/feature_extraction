function D = histogram_similarity(coch_orig, coch_synth, P)

% Measures the similarity of the histograms between cochlear channels and the
% cochlear channels of filtered cocleograms. Similarity is measured using the
% Jensen-Shannon divergence and the Bhattacharyya coefficient. See jsdiv.m and
% bhat_coef.m.
%
% -- Inputs --
%
% coch_orig: cochleogram for the original natural sound
%
% coch_synth: cochleogram for the synthetic
%
% P: parameter structure, see synthesis_parameters_default.m
%
% 2016-08-31: Created, Sam NH

% number of bins to use to compute the histogram
n_bins = [20 50];

%% Histogram similarity of cochlear channels

clear D;
for i = 1:length(n_bins)
    
    % Jensen-Shannon divergence
    D.(['coch_jsdiv_' num2str(n_bins(i)) 'bins']) ...
        = jsdiv(coch_orig, coch_synth, 'n_bins', n_bins(i))';
    
    % Bhattacharyya coefficient
    D.(['coch_bhatcoef_' num2str(n_bins(i)) 'bins']) ...
        = bhat_coef(coch_orig, coch_synth, 'n_bins', n_bins(i))';
    
end

%% Histogram similarity of filtered cochleograms

% temporal modulation filters
pos_temp_mod_rates = P.temp_mod_rates(P.temp_mod_rates>0);
D = hist_similarity_filtcoch(...
    coch_orig, coch_synth, NaN, pos_temp_mod_rates, ...
    P, n_bins, D, 'temp_mod');

% spectral modulation filters
D = hist_similarity_filtcoch(...
    coch_orig, coch_synth, P.spec_mod_rates, NaN, ...
    P, n_bins, D, 'spec_mod');

% spectrotemporal modulation filters
temp_mod_rates_pos_and_neg = [pos_temp_mod_rates, -pos_temp_mod_rates];
D = hist_similarity_filtcoch(...
    coch_orig, coch_synth, P.spec_mod_rates, temp_mod_rates_pos_and_neg, ...
    P, n_bins, D, 'spectemp_mod');

function D = hist_similarity_filtcoch(coch_orig, coch_synth, ...
    spec_mod_rates, temp_mod_rates, P, n_bins, D, prefix)

% Helper function for computing histogram similarity from filtered cochleograms

% pad cochleogram
coch_orig_padded = pad_coch(coch_orig, P);
coch_synth_padded = pad_coch(coch_synth, P);

% fourier transforms of padded cochleograms
FT_coch_orig = fft2(coch_orig_padded);
FT_coch_synth = fft2(coch_synth_padded);

% dimensions of padded cochleogram
[T,F] = size(coch_synth_padded);
assert(all(size(coch_orig_padded) == size(coch_synth_padded)));

n_spec_mod_rates = length(spec_mod_rates);
n_temp_mod_rates = length(temp_mod_rates);

% initialize
for k = 1:length(n_bins)
    D.([prefix '_bhatcoef_' num2str(n_bins(k)) 'bins']) = ...
        nan(n_spec_mod_rates, n_temp_mod_rates, size(coch_orig,2));
    D.([prefix '_jsdiv_' num2str(n_bins(k)) 'bins']) = ...
        nan(n_spec_mod_rates, n_temp_mod_rates, size(coch_orig,2));
end

% loop through all filters
for i = 1:length(spec_mod_rates)
    for j = 1:length(temp_mod_rates)
        
        % 2D filter
        Hts = filt_spectemp_mod(...
            spec_mod_rates(i), temp_mod_rates(j), F, T, P);
        
        % apply filter
        filtcoch_orig_padded = real(ifft2(FT_coch_orig .* Hts));
        filtcoch_synth_padded = real(ifft2(FT_coch_synth .* Hts));
        
        % remove padding
        filtcoch_orig = remove_pad(filtcoch_orig_padded, P);
        filtcoch_synth = remove_pad(filtcoch_synth_padded, P);
        
        for k = 1:length(n_bins)
            
            % Jensen-Shannon divergence
            D.([prefix '_jsdiv_' num2str(n_bins(k)) 'bins'])(i,j,:) ...
                = jsdiv(filtcoch_orig, filtcoch_synth, 'n_bins', n_bins(k));
            
            % Bhattacharyya coefficient
            D.([prefix '_bhatcoef_' num2str(n_bins(k)) 'bins'])(i,j,:) ...
                = bhat_coef(filtcoch_orig, filtcoch_synth, 'n_bins', n_bins(k));
            
        end
    end
end

% remove singleton dimensions
measures = {'jsdiv', 'bhatcoef'};
for k = 1:length(n_bins);
    for i = 1:length(measures)
        try
            if n_spec_mod_rates == 1
                D.([prefix '_' measures{i} '_' num2str(n_bins(k)) 'bins']) = squeeze_dim(...
                    D.([prefix '_' measures{i} '_' num2str(n_bins(k)) 'bins']), 1);
            end
            
            if n_temp_mod_rates == 1
                D.([prefix '_' measures{i} '_' num2str(n_bins(k)) 'bins']) = squeeze_dim(...
                    D.([prefix '_' measures{i} '_' num2str(n_bins(k)) 'bins']), 2);
            end
        catch
            keyboard
        end
    end
end