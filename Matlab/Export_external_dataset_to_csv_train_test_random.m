
chunks = 1;
version = 'v5';
numReps = 5;
trainPercent = 0.8;

labelRepIncrement = 0; %Top avoid overwrite previous files. Default= 0

pdpSizes = [3,7,15,30];
for ii=1:1:numReps
    for jj=1:1:length(pdpSizes)
        pdpSize = pdpSizes(jj);
        pathFeatures = ['./Measurements/External/RangingWithCIRData3_' version '_features_' version '_pdp_' num2str(pdpSize) '.mat'];
   
        useNormalized=0
        if (jj==1)
           % Split set in training and test
            % 50% of los/nlos in each set
            [shuffleIndexes, trainRows, testRows ]=do_export_external_dataset_to_csv_train_test_random_v5(pathFeatures, chunks, ['Rand_' num2str(ii+labelRepIncrement) '_pdp_' num2str(pdpSize) '_'], useNormalized,trainPercent);
        else
            do_export_external_dataset_to_csv_train_test_random_v5(pathFeatures, chunks, ['Rand_' num2str(ii+labelRepIncrement) '_pdp_' num2str(pdpSize) '_'], useNormalized, trainPercent,shuffleIndexes, trainRows, testRows);
        end
        
        useNormalized=1
        do_export_external_dataset_to_csv_train_test_random_v5(pathFeatures, chunks, ['Rand_' num2str(ii+labelRepIncrement) '_pdp_' num2str(pdpSize) '_'], useNormalized,trainPercent, shuffleIndexes,trainRows, testRows);
    end
end






