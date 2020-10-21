function [shuffleRows, rowsTrain, rowsTest]=do_export_external_dataset_to_csv_train_test_random_v5(pathFeatures, chunks, prefix, useNormalized, trainPercent, shuffleRowsExplicit, rowsTrainExplicit, rowsTestExplicit)

    theFeatures = load(pathFeatures);

     nlos_all = theFeatures.features.nlos;
     rss_all = theFeatures.features.rss;
     range_all = theFeatures.features.range;
     energy_all = theFeatures.features.energy;
     mean_delay_all = theFeatures.features.mean_delay;
     rms_delay_all = theFeatures.features.rms_delay;
     label_normalized = '';
     if (useNormalized==1)
         pdp_resampled_all = cell2mat(theFeatures.features.pdp_downsampled_normalized_no_noise(1:end));
         cir_first_all = cell2mat(theFeatures.features.cir_152_normalized_no_noise(1:end));
         cir_all = cell2mat(theFeatures.features.cir_normalized_no_noise(1:end)); 
         label_normalized ='_normalized';
     else
         pdp_resampled_all = cell2mat(theFeatures.features.pdp_downsampled(1:end));
         cir_first_all = cell2mat(theFeatures.features.cir_152(1:end));
         cir_all = cell2mat(theFeatures.features.cir(1:end));       
     end


     rss_all = normalize(rss_all);
     range_all = normalize(range_all);
     energy_all = normalize(energy_all);
     mean_delay_all = normalize(mean_delay_all);
     rms_delay_all = normalize(rms_delay_all); 

     
     if ~exist('shuffleRowsExplicit','var')
         %shuffleRows = randperm(size(nlos_all,1));
         shuffleRows =  Shuffle(1:size(nlos_all,1));
     else
         shuffleRows= shuffleRowsExplicit;
     end
     
     
     nlos_all = nlos_all(shuffleRows,:);  
     rss_all = rss_all(shuffleRows,:);  
     range_all = range_all(shuffleRows,:);  
     energy_all = energy_all(shuffleRows,:);  
     mean_delay_all =mean_delay_all(shuffleRows,:);  
     rms_delay_all =rms_delay_all(shuffleRows,:);  
     pdp_resampled_all = pdp_resampled_all(shuffleRows,:);  
     cir_first_all = cir_first_all(shuffleRows,:);
     cir_all = cir_all(shuffleRows,:);  

     numMeasurements = size(nlos_all,1);
     
     
     %Split Train and Test
     
     if ~exist('rowsTrainExplicit','var')
         numMeasurementsTrain = floor(numMeasurements * trainPercent);
         numMeasurementsTrainLOS = floor(0.5*numMeasurementsTrain);
         numMeasurementsTrainNLOS = numMeasurementsTrain - numMeasurementsTrainLOS;

         secuence = 1:1:size(nlos_all,1);

        losIndexes = secuence(nlos_all==0);
        nlosIndexes = secuence(nlos_all==1);

        losTrainIndexes = losIndexes(1:numMeasurementsTrainLOS);
        nlosTrainIndexes = nlosIndexes(1:numMeasurementsTrainNLOS);
        rowsTrain = mergesorted(losTrainIndexes, nlosTrainIndexes);

        losTestIndexes = losIndexes(numMeasurementsTrainLOS+1:end);
        nlosTestIndexes = nlosIndexes(numMeasurementsTrainNLOS+1:end);
        rowsTest = mergesorted(losTestIndexes, nlosTestIndexes);
     else
         rowsTrain= rowsTrainExplicit;
         rowsTest = rowsTestExplicit;
     end
     
         
     train_nlos_all = nlos_all(rowsTrain,:);  
     train_rss_all = rss_all(rowsTrain,:);  
     train_range_all = range_all(rowsTrain,:);  
     train_energy_all = energy_all(rowsTrain,:);  
     train_mean_delay_all =mean_delay_all(rowsTrain,:);  
     train_rms_delay_all =rms_delay_all(rowsTrain,:);  
     train_pdp_resampled_all = pdp_resampled_all(rowsTrain,:);  
     train_cir_first_all = cir_first_all(rowsTrain,:);
     train_cir_all = cir_all(rowsTrain,:);  
     
     test_nlos_all = nlos_all(rowsTest,:);  
     test_rss_all = rss_all(rowsTest,:);  
     test_range_all = range_all(rowsTest,:);  
     test_energy_all = energy_all(rowsTest,:);  
     test_mean_delay_all =mean_delay_all(rowsTest,:);  
     test_rms_delay_all =rms_delay_all(rowsTest,:);  
     test_pdp_resampled_all = pdp_resampled_all(rowsTest,:);  
     test_cir_first_all = cir_first_all(rowsTest,:);
     test_cir_all = cir_all(rowsTest,:);  
     
     
     %Saving Training
     
     numMeasurementsTrain = size(train_nlos_all,1);
     numMeasurementsTrainPerChunk = floor(numMeasurementsTrain/chunks);
     
    for ii=1:1:chunks 

        start = (ii-1)*numMeasurementsTrainPerChunk + 1;
        if (ii==chunks)
            %Last chunk
            limit=size(train_nlos_all,1);
        else
            limit= (ii-1)*numMeasurementsTrainPerChunk +numMeasurementsTrainPerChunk;
        end
         nlos = train_nlos_all(start:limit);
         rss = train_rss_all(start:limit);
         range = train_range_all(start:limit);
         energy = train_energy_all(start:limit);
         mean_delay = train_mean_delay_all(start:limit);
         rms_delay = train_rms_delay_all(start:limit);
         pdp_resampled = train_pdp_resampled_all(start:limit,:);
         cir_first = train_cir_first_all(start:limit,:);
         cir = train_cir_all(start:limit,:);

         T = table(nlos, rss, range, energy, mean_delay, rms_delay, pdp_resampled, cir_first, cir);
         writetable(T,[prefix 'External_cir_and_pdp_set_3_TRAIN_' label_normalized '_' num2str(ii) '.csv'],'Delimiter',',','QuoteStrings',true);
    end

    %Saving Test
     
     numMeasurementsTest = size(test_nlos_all,1);
     numMeasurementsTestPerChunk = floor(numMeasurementsTest/chunks);
     

    for ii=1:1:chunks 

        start = (ii-1)*numMeasurementsTestPerChunk + 1;
        if (ii==chunks)
            %Last chunk
            limit=size(test_nlos_all,1);
        else
            limit= (ii-1)*numMeasurementsTestPerChunk +numMeasurementsTestPerChunk;
        end
        
         nlos = test_nlos_all(start:limit);
         rss = test_rss_all(start:limit);
         range = test_range_all(start:limit);
         energy = test_energy_all(start:limit);
         mean_delay = test_mean_delay_all(start:limit);
         rms_delay = test_rms_delay_all(start:limit);
         pdp_resampled = test_pdp_resampled_all(start:limit,:);
         cir_first = test_cir_first_all(start:limit,:);
         cir = test_cir_all(start:limit,:);

         T = table(nlos, rss, range, energy, mean_delay, rms_delay, pdp_resampled, cir_first, cir);
         writetable(T,[prefix 'External_cir_and_pdp_set_3_TEST_' label_normalized '_' num2str(ii) '.csv'],'Delimiter',',','QuoteStrings',true);
    end
    
    function result = normalize(vector)
        minValue = min(vector);
        maxValue = max(vector);
        result = (vector - ones(length(vector),1).*minValue)./(ones(length(vector),1).*maxValue - ones(length(vector),1).*minValue);
    end

end



