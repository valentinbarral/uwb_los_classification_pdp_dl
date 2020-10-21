function extractFeaturesFromCir(version,fileName,pdp_downsampled_out,pdp_downsampling_factor)
%extractFeaturesFromCir Gets PDP from CIR measurements

% References:
% [1] Valentín Barral Vales, "ULTRA WIDEBAND LOCATION IN SCENARIOS WITHOUT
%     CLEAR LINE OF SIGHT: A PRACTICAL APPROACH", Doctoral Thesis, 2020.
%
% [2] Stefano Maranò, Wesley M. Gifford, Henk Wymeersch, Moe Z. Win,
%     "NLOS Identification and Mitigation for Localization Based on UWB
%     Experimental Data", IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS,
%     VOL. 28, NO. 7, SEPTEMBER 2010.


%% Configuration


% Measurement data
measfiles = {[fileName version]};

% Settings for the rise time feature (see reference [2])
alpha = 6; % alpha parameter to calculate the rise time
beta = 0.6; % beta parameters to calculare the rise time


%% Code
    
for ii = 1:numel(measfiles)
    file = measfiles{ii};
    
    % extract struct array with the measurement data
    measdata = load(file);
    measdata = struct2cell(measdata);
    measdata = measdata{1};
    
    nmeas = numel(measdata.distance); % number of measurements
    
    % variables to save the features
    cir_cell = cell(nmeas,1); % CIRs
    cir_original_cell = cell(nmeas,1);
    cir_original_normalized_cell = cell(nmeas,1);
    cir_original_normalized_no_noise_cell = cell(nmeas,1);
    
    pdp_downsampled_cell = cell(nmeas,1); % Downsampled PDP
    pdp_downsampled_normalized_cell = cell(nmeas,1); % Downsampled PDP
    pdp_downsampled_normalized_no_noise_cell = cell(nmeas,1);
    
    cir_152_cell = cell(nmeas,1); % 152 first samples of the CIR
    cir_152_normalized_cell = cell(nmeas,1); % 152 first samples of the CIR
    cir_152_normalized_no_noise_cell = cell(nmeas,1);

    energy_vec = zeros(nmeas,1); % Energy of the CIR (not including the noise)
    energy_152_vec = zeros(nmeas,1); % Energy of the first 152 CIR samples (including the noise)
    max_amplitude_vec = zeros(nmeas,1); % Maximum amplitude
    t_rise_vec = zeros(nmeas,1); % Rise time
    mean_delay_vec = zeros(nmeas,1); % Mean excess delay
    rms_delay_vec = zeros(nmeas,1); % Root mean square delay
    kurtosis_vec = zeros(nmeas,1); % Kurtosis
    
    % Mask of valid measurements: we remove too noisy measurements
    valid_meas_mask = false(nmeas, 1); 
            
    meastimer = tic();
    fprintf('Processing %s... ', file);
    
    % print simple progress indicator
    progressmsg = sprintf('[%3d/%3d](%5.2f%%)', 0, nmeas, 0);
    fprintf('%s', progressmsg);
        
    start_tomas = zeros(nmeas,1);
    start_measurements = zeros(nmeas,1);
    % process measurements
    for jj = 1:nmeas
        % get CIR
        % we discard the 4 first CIR samples as said in the Decawave 
        % documentation. It seems that for some unknown reason the last 2
        % samples are [0 NaN] in most of the cases so we also discard them
        cir = measdata.cir(jj,5:end-2);
        cir_original_cell{jj} = cir;
        
        % estimate noise power
        noise_tmax = 500;
        assert(noise_tmax > 1);
        noise_pw_est = mean(abs(cir(1:noise_tmax)).^2);

        % the first 600~700 samples of the CIR are only noise
        % here we discard the first 500 samples of the CIR
        cir_ndiscard = 500;
        cir = cir(cir_ndiscard+1:end);
        cir_cell{jj} = cir;

        % max absolute value and corresponding index
        [max_amplitude, cir_max_i] = max(abs(cir));

        % CIR energy
        energy = sum(abs(cir).^2 - noise_pw_est);
        energy_vec(jj) = energy;
        
        if energy <= 0
            % If this happens the measurent is too noisy
            continue;
        end

        % maximum amplitude
        max_amplitude_vec(jj) = max_amplitude;

        % rise time
        t_high = find(abs(cir) >= beta*max_amplitude, 1, 'first');
        t_low = find(abs(cir) >= alpha*sqrt(noise_pw_est), 1, 'last');
        
        t_high_cell{jj} = t_high;
        t_low_cell{jj} = t_low;
        
        if isempty(t_low) || isempty(t_high)
            t_rise_vec(jj) = nan;
            continue;
        else
            t_rise_vec(jj) = t_low - t_high;
        end

        % create variables t_start and t_end to select CIR samples
        t_start = max(1, t_high - 3);
        t_end = t_start + t_low - t_high + 1;
        t_start_measurement = measdata.fpindex(jj) - cir_ndiscard;
        
        start_tomas(jj) = t_start;
        start_measurements(jj) = t_start_measurement;
        if t_end > numel(cir)
            continue;
        end
        
        % Downsampled PDP
        % the downsampled PDP is calculated starting at t_high         
        pdp_start_t_cell{jj} = t_start;

        cir1 = cir(t_start:end);
        cir1 = reshape(cir1(1:pdp_downsampling_factor*pdp_downsampled_out), ...
                       [], pdp_downsampled_out);

        pdp_downsampled = mean(abs(cir1).^2);
        pdp_downsampled_cell{jj} = pdp_downsampled;
        
        % 152 CIR samples from the first path
        cir_152 = cir(t_start + (0:151));
        cir_152_cell{jj} = cir_152;
        energy_152 = sum(abs(cir_152).^2);
        energy_152_vec(jj) = energy_152;
        
        % normalized CIR
        cir_original_normalized_cell{jj} = ...
            cir_original_cell{jj}/sqrt(energy_vec(jj));
        
        cir_t_rise_mask = false(1, numel(cir_original_cell{jj}));
        cir_t_rise_mask(cir_ndiscard + (t_start:t_end)) = true;
        
        cir_original_normalized_no_noise_cell{jj} = cir_original_normalized_cell{jj};
        cir_original_normalized_no_noise_cell{jj}(~cir_t_rise_mask) = 0;
        
        
        % normalized CIR first 152 samples
        cir_152_normalized_cell{jj} = ...
            cir_152_cell{jj}/sqrt(energy_vec(jj));
        
        cir_152_t_rise_mask = false(1, 152);
        cir_152_t_rise_mask(1:(t_end - t_start + 1)) = true;
        
        cir_152_normalized_no_noise_cell{jj} = cir_152_normalized_cell{jj};
        cir_152_normalized_no_noise_cell{jj}(~cir_152_t_rise_mask) = 0;
 
        
        % normalized downsampled PDP
        cir1 = cir(t_start:end)/sqrt(energy_vec(jj));
        cir1 = reshape(cir1(1:pdp_downsampling_factor*pdp_downsampled_out), ...
                       [], pdp_downsampled_out);
                   
        pdp_downsampled = mean(abs(cir1).^2);
        pdp_downsampled_normalized_cell{jj} = pdp_downsampled;
        
        % normalized downsampled PDP without noise
        cir1 = cir_original_normalized_no_noise_cell{jj}((cir_ndiscard + t_start):end);
        cir1 = reshape(cir1(1:pdp_downsampling_factor*pdp_downsampled_out), ...
                       [], pdp_downsampled_out);
            
        pdp_downsampled = mean(abs(cir1).^2);
        pdp_downsampled_normalized_no_noise_cell{jj} = pdp_downsampled;

        % mean excess delay
        t_152 = t_high + (0:151);
        mean_delay = sum(t_152.*(abs(cir_152).^2)./energy_152);
        mean_delay_vec(jj) = mean_delay;
        
        % RMS delay excess
        rms_delay_vec(jj) =...
            sqrt(sum((t_152 - mean_delay).^2.*(abs(cir_152).^2)/energy_152));

        % kurtosis
        kurtosis_vec(jj) =...
            sum((t_152 - mean_delay).^4.*(abs(cir_152).^2)/energy_152)./...
            sum((t_152 - mean_delay).^2.*(abs(cir_152).^2)/energy_152).^2;

        % mark measurement as valid
        valid_meas_mask(jj) = true;
        
        progressmsglen = numel(progressmsg);
        progressmsg = sprintf('[%3d/%3d](%5.2f%%)', jj, nmeas, 100*jj/nmeas);
        fprintf([repmat('\b', 1, progressmsglen), '%s'], progressmsg);
    end
    fprintf(' OK (%g s)\n', toc(meastimer));
    
    % create struct with the results
    features.cir = cir_original_cell(valid_meas_mask);
    features.cir_normalized = cir_original_normalized_cell(valid_meas_mask);
    features.cir_normalized_no_noise = cir_original_normalized_no_noise_cell(valid_meas_mask);
    
    features.pdp_downsampled = pdp_downsampled_cell(valid_meas_mask);
    features.pdp_downsampled_normalized = pdp_downsampled_normalized_cell(valid_meas_mask);
    features.pdp_downsampled_normalized_no_noise = pdp_downsampled_normalized_no_noise_cell(valid_meas_mask);

    features.cir_152 = cir_152_cell(valid_meas_mask);
    features.cir_152_normalized = cir_152_normalized_cell(valid_meas_mask);
    features.cir_152_normalized_no_noise = cir_152_normalized_no_noise_cell(valid_meas_mask);

    features.energy = energy_vec(valid_meas_mask);
    features.energy_152 = energy_152_vec(valid_meas_mask);
    features.max_amplitude = max_amplitude_vec(valid_meas_mask);
    features.t_rise = t_rise_vec(valid_meas_mask);
    features.mean_delay = mean_delay_vec(valid_meas_mask);
    features.rms_delay = rms_delay_vec(valid_meas_mask);
    features.kurtosis = kurtosis_vec(valid_meas_mask);
    
    features.range = measdata.range(valid_meas_mask);
    features.distance = measdata.distance(valid_meas_mask);
    features.rss = measdata.rss(valid_meas_mask);
    features.nlos = measdata.nlos(valid_meas_mask);
    features.channel = measdata.channel(valid_meas_mask);
    
    % save struct to file
    file_path=strsplit(file,filesep);
    file_name=file_path{end};
    save([file_name '_features_' version '_pdp_' num2str(pdp_downsampled_out)], 'features');

end

