
version = 'v5';
fileName = './Measurements/External/RangingWithCIRData3_';

pdp_downsampled_out = 3;
pdp_downsampling_factor = 40;
extractFeaturesFromCir(version, fileName, pdp_downsampled_out, pdp_downsampling_factor);

pdp_downsampled_out = 7;
pdp_downsampling_factor = 20;
extractFeaturesFromCir(version, fileName, pdp_downsampled_out, pdp_downsampling_factor);

pdp_downsampled_out = 15;
pdp_downsampling_factor = 10;
extractFeaturesFromCir(version, fileName, pdp_downsampled_out, pdp_downsampling_factor);

pdp_downsampled_out = 30;
pdp_downsampling_factor = 5;
extractFeaturesFromCir(version, fileName, pdp_downsampled_out, pdp_downsampling_factor);