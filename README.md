# Using the Power Delay Profile to accelerate the training of neural network-based classifiers for the identification of LOS and NLOS UWB propagation conditions


This repository contains the code needed to replicate the experiments described in the article : "Using the Power Delay Profile to accelerate the training of neural network-based classifiers for the identification of LOS and NLOS UWB propagation conditions". The code is divided into two parts, one in Matlab to pre-process the samples and extract the PDP, and another part in Python+Tensorflow to train and test the LOS-NLOS classifier.

## Pre-processing in Matlab

To generate measurement sets be used later in Tensorflow, it is necessary to pre-process the original data. This requires the following steps:

- Clone the repository [https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set) inside the "./Matlab/Measurements/ directory.

- Run the file "./Matlab/parseNLOSClassificationData.m". This will generate a file called "./Measurements/External/RangingWithCIRData3_v5.mat".

- Run the file "./Matlab/extractFeaturesFromCir.m". This will generate four new files with the PDP samples and the rest of the features.

- Run the file "./Matlab/Export_external_dataset_to_csv_train_test_random.m". This will generate random sets of training and testing. The number of generated sets is configured within the script in the variable "numReps".

- Finally, the generated ".csv" files must be copied into the "./ExternalDatasetWithPDP_v5" folder

## Classification

To run the simulation, the "main.py" file must be executed (it is necessary to have previously installed the [Tensorflow](https://www.tensorflow.org/install) libraries).

The results are stored in "./Results_v5". Different figures can be obtained using the scripts "./plot_results_multi_pdp.py" and "plot_results.py". 

### References

This work uses the UWB measurements datataset related with the paper: 

Klemen Bregar, Andrej Hrovat, Mihael Mohorčič, "NLOS Channel Detection with Multilayer Perceptron in Low-Rate Personal Area Networks for Indoor Localization Accuracy Improvement". Proceedings of the 8th Jožef Stefan International Postgraduate School Students’ Conference, Ljubljana, Slovenia, May 31-June 1, 2016.
