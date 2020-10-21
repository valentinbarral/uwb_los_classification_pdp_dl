# MIT License

# Copyright (c) 2020 Group of Electronic Technology and Communications. University of A Coruna.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import pandas as pd
from numpy import vstack
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, DenseFeatures, Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Reshape, MaxPooling1D, LSTM
import glob
import functools
import time
from tensorflow.keras import backend as K
import pydotplus
import collections

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_dataset(file_path, **kwargs):
    #batch_size=batch_size
    dataset = tf.data.experimental.make_csv_dataset(file_path,label_name=LABEL_COLUMN, batch_size=batch_size,na_value="?",num_epochs=1,ignore_errors=True, shuffle=False, header=True, **kwargs)
    return dataset

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))

def pack(features, label):
      return tf.stack([float(i) for i in list(features.values())], axis=-1), label

def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# DATASET_SIZE=41996
batch_size = 64
epochs = 20
numReps =10
cir_energy_mode = 0  # 0:raw, 1:normalized
version = 'v5'

# Variant: 
# all: test all modes with all pdp sizes 
# all_no_extra: test only modes without extra features, but with all pdp sizes
# all_only_extra: test only modes with extra features, but with all pdp sizes
# all_single_pdp: test all modes but only one pdp size. The pdp size is the first of the array pdpLFactors
# pdps: test only the modes that use pdp, with all the pdp sizes
# all_only_extra_single_pdp: test only modes with extra features and only one PDP size.
variant = 'all'

pdpLFactors = [5,10,20,40]
modesToTest = [0,1,2,3,4,5] #0:CIR, 1:CIR152, 2:PDP, 3:CIR + EXTRA, 4:CIR152 + EXTRA, 5:PDP + EXTRA
if variant== 'all':
    modesToTest = [0,1,2,3,4,5]
elif variant == 'all_no_extra':
    modesToTest = [0,1,2]
elif variant == 'all_only_extra':
    modesToTest = [3,4,5]
elif variant == 'all_single_pdp':
    modesToTest = [0,1,2,3,4,5]
    pdpLFactors = [pdpLFactors[0]]
elif variant == 'all_only_extra_single_pdp':
    modesToTest = [3,4,5]
    pdpLFactors = [pdpLFactors[0]]
elif variant == 'pdps':
    modesToTest = [2,5]   



modeStr = ['cir', 'cir[152]', 'pdp', 'others+cir', 'others+cir[152]', 'others+pdp' ]
cir_size = 1010

cir_first_size = 152
LABEL_COLUMN = 'nlos'
LABELS = [0, 1]

cir_energy_mode_label = ''
if cir_energy_mode==1:
    cir_energy_mode_label = '_normalized'


for mode in modesToTest: #0:CIR, 1:CIR152, 2:PDP, 3:CIR + EXTRA, 4:CIR152 + EXTRA, 5:PDP + EXTRA
    usesPdp = False
    pdpLFactorsToTest = [5]
    if ((2 == mode) or (5 == mode)):
        #WE test different PDP values
        usesPdp = True
        pdpLFactorsToTest = pdpLFactors
        
    for pdp_factor in pdpLFactorsToTest:
        pdp_size = int(cir_first_size/pdp_factor)

        print('++++++++++++++ START +++++++++++++++++')
        print('MODE: ' + modeStr[mode])
        if usesPdp:
            print('PDP L Factor: ' + str(pdp_factor))
            print('PDP Num. Samples: ' + str(pdp_size))
        print('Num Reps: ' + str(numReps))
        print('......................................')
        print('......................................')

        resultsAccuracy = []
        resultsExecutioinTime = []
        resultsF1 = []
        resultsPrecision = []
        resultsRecall = []
        

        if mode==0:
            #ONLY CIR
            SELECT_COLUMNS_CIR = ['nlos']

            for x in range(cir_size):
                SELECT_COLUMNS_CIR.append('cir_' + str(x+1))
            SELECT_COLUMNS = [] + SELECT_COLUMNS_CIR
            inputSizeNoCir =0
            inputSizeCir = cir_size
        elif mode==1:
            #first 152 cir
            SELECT_COLUMNS_CIR = ['nlos']
            for x in range(cir_first_size):
                SELECT_COLUMNS_CIR.append('cir_first_' + str(x+1))
            SELECT_COLUMNS = [] + SELECT_COLUMNS_CIR
            inputSizeNoCir =0
            inputSizeCir = cir_first_size 
        elif mode==2:
            #ONLY pdp
            SELECT_COLUMNS_CIR = ['nlos']
            for x in range(pdp_size):
                SELECT_COLUMNS_CIR.append('pdp_resampled_' + str(x+1))
            SELECT_COLUMNS = [] + SELECT_COLUMNS_CIR
            inputSizeNoCir =0
            inputSizeCir = pdp_size    
        elif mode==3:
            #Others and cir
            SELECT_COLUMNS_CIR = []
            #SELECT_COLUMNS_NO_CIR = ['rss', 'range','energy','mean_delay','rms_delay']
            SELECT_COLUMNS_NO_CIR = ['range','energy']
            others_size = len(SELECT_COLUMNS_NO_CIR)
            SELECT_COLUMNS_CIR = []
            for x in range(cir_size):
                SELECT_COLUMNS_CIR.append('cir_' + str(x+1))
            SELECT_COLUMNS = ['nlos'] + SELECT_COLUMNS_NO_CIR + SELECT_COLUMNS_CIR
            SELECT_COLUMNS_NO_CIR =  ['nlos'] + SELECT_COLUMNS_NO_CIR
            SELECT_COLUMNS_CIR =  ['nlos'] + SELECT_COLUMNS_CIR
            inputSizeNoCir =others_size
            inputSizeCir = cir_size   
        elif mode==4:
            #first 152 cir + extra
            SELECT_COLUMNS_CIR = []
            #SELECT_COLUMNS_NO_CIR = ['rss', 'range','energy','mean_delay','rms_delay']
            SELECT_COLUMNS_NO_CIR = ['range','energy']
            others_size = len(SELECT_COLUMNS_NO_CIR)
            SELECT_COLUMNS_CIR = []
            for x in range(cir_first_size):
                SELECT_COLUMNS_CIR.append('cir_first_' + str(x+1))
            SELECT_COLUMNS = ['nlos'] + SELECT_COLUMNS_NO_CIR + SELECT_COLUMNS_CIR
            SELECT_COLUMNS_NO_CIR =  ['nlos'] + SELECT_COLUMNS_NO_CIR
            SELECT_COLUMNS_CIR =  ['nlos'] + SELECT_COLUMNS_CIR
            inputSizeNoCir =others_size
            inputSizeCir = cir_first_size     
        elif mode==5:
                #PDP + EXTRA
            SELECT_COLUMNS_CIR = []
            #SELECT_COLUMNS_NO_CIR = ['rss', 'range','energy','mean_delay','rms_delay']
            SELECT_COLUMNS_NO_CIR = ['range','energy']
            others_size = len(SELECT_COLUMNS_NO_CIR)
            for x in range(pdp_size):
                SELECT_COLUMNS_CIR.append('pdp_resampled_' + str(x+1))
            SELECT_COLUMNS = ['nlos'] + SELECT_COLUMNS_NO_CIR + SELECT_COLUMNS_CIR
            SELECT_COLUMNS_NO_CIR =  ['nlos'] + SELECT_COLUMNS_NO_CIR
            SELECT_COLUMNS_CIR =  ['nlos'] + SELECT_COLUMNS_CIR
            inputSizeNoCir =others_size
            inputSizeCir = pdp_size

 

                    
        train_only_with_cir = mode in [0,1,2]

        # numBatchs = int(DATASET_SIZE/batch_size)
        # trainValSize = int(0.8 * numBatchs)
        # train_size = int(0.9 * trainValSize)

        feature_columns_cir = []
        for header in SELECT_COLUMNS_CIR:
            feature_columns_cir.append(tf.feature_column.numeric_column(header))

        if (not train_only_with_cir):
            feature_columns_no_cir = []
            for header in SELECT_COLUMNS_NO_CIR:
                feature_columns_no_cir.append(tf.feature_column.numeric_column(header))

        for ii in range(numReps):

            print('### Rep: ' + str(ii+1) + ' of ' + str(numReps))


            train_file_path = os.path.join(os.path.dirname("./"), 'ExternalDatasetWithPDP_'+ version +'/'+'Rand_' + str(ii+1)  + '_pdp_' + str(pdp_size) +'_External_cir_and_pdp_set_3_TRAIN_'+ cir_energy_mode_label+'_1.csv')
            #full_dataset_only_cir = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
            train_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
            #show_batch(train_dataset)
           
            test_file_path = os.path.join(os.path.dirname("./"), 'ExternalDatasetWithPDP_'+ version +'/'+'Rand_' + str(ii+1)  + '_pdp_' + str(pdp_size) +'_External_cir_and_pdp_set_3_TEST_'+ cir_energy_mode_label+'_1.csv')
            test_dataset = get_dataset(test_file_path, select_columns=SELECT_COLUMNS)
            #show_batch(test_dataset)
            
           # train_dataset, test_dataset= split_dataset(full_dataset_only_cir,0.2)
            #train_dataset, val_dataset = split_dataset(train_dataset,0.1)

            def select_features(cols):
                def select_cols(features,label):
                    # print(features.keys())
                    # print('...')
                    # print(label)
                    key = features.keys()
                    r = collections.OrderedDict()
                    for key in features.keys():
                        if key in cols:
                            r[key]=features[key]

                    return r,label
                return select_cols

            # TRAIN DATASET
            train_dataset_cir = train_dataset.map(select_features(SELECT_COLUMNS_CIR))
            if (not train_only_with_cir):
                train_dataset_no_cir = train_dataset.map(select_features(SELECT_COLUMNS_NO_CIR))

            packed_train_dataset_cir = train_dataset_cir.map(pack)
            if (not train_only_with_cir):
                packed_train_dataset_no_cir = train_dataset_no_cir.map(pack)
                train_dataset_all_zip = tf.data.Dataset.zip((packed_train_dataset_cir, packed_train_dataset_no_cir))
                train_dataset_all_X = train_dataset_all_zip.map(lambda x1, x2: {'input_1': x1[0], 'input_2': x2[0]})
                train_dataset_all_Y = train_dataset_all_zip.map(lambda x1, x2: x2[1])
                train_dataset_all = tf.data.Dataset.zip((train_dataset_all_X, train_dataset_all_Y))
            else:
                train_dataset_all =packed_train_dataset_cir

            # VALIDATION DATASET

            # val_dataset_cir = val_dataset.map(select_features(SELECT_COLUMNS_CIR))
            # if (not train_only_with_cir):
            #     val_dataset_no_cir = val_dataset.map(select_features(SELECT_COLUMNS_NO_CIR))

            # packed_val_dataset_cir = val_dataset_cir.map(pack)
            # if (not train_only_with_cir):
            #     packed_val_dataset_no_cir = val_dataset_no_cir.map(pack)
            #     val_dataset_zip = tf.data.Dataset.zip((packed_val_dataset_cir, packed_val_dataset_no_cir))
            #     val_dataset_all_X = val_dataset_zip.map(lambda x1, x2: {'input_1': x1[0], 'input_2': x2[0]})
            #     val_dataset_all_Y = val_dataset_zip.map(lambda x1, x2: x1[1])
            #     val_dataset_all = tf.data.Dataset.zip((val_dataset_all_X, val_dataset_all_Y))
            # else:
            #     val_dataset_all = packed_val_dataset_cir

            # TEST DATASET
            test_dataset_cir = test_dataset.map(select_features(SELECT_COLUMNS_CIR))
            if (not train_only_with_cir):
                test_dataset_no_cir = test_dataset.map(select_features(SELECT_COLUMNS_NO_CIR))

            packed_test_dataset_cir = test_dataset_cir.map(pack)
            if (not train_only_with_cir):
                packed_test_dataset_no_cir = test_dataset_no_cir.map(pack)
                test_dataset_zip = tf.data.Dataset.zip((packed_test_dataset_cir, packed_test_dataset_no_cir))
                test_dataset_all_X = test_dataset_zip.map(lambda x1, x2: {'input_1': x1[0], 'input_2': x2[0]})
                test_dataset_all_Y = test_dataset_zip.map(lambda x1, x2: x1[1])
                test_dataset_all = tf.data.Dataset.zip((test_dataset_all_X, test_dataset_all_Y))
            else:
                test_dataset_all = packed_test_dataset_cir

            # FIRST INPUT (CIR RELATED FEATURES)
            model1 = Sequential()
            if (not train_only_with_cir):
                model1.add(tf.keras.layers.Input(shape=(inputSizeCir,),name='input_1'))
            else:
                model1.add(tf.keras.layers.Input(shape=(inputSizeCir,)))

            model1.add(Reshape((1,inputSizeCir)))
            model1.add(Conv1D(10, 4, padding='same',  activation='relu', input_shape=(inputSizeCir,)))
            model1.add(BatchNormalization())
            model1.add(Conv1D(20, 5, padding='same',  activation='relu'))
            model1.add(BatchNormalization())
            model1.add(MaxPooling1D(2,strides=2, padding='same'))
            model1.add(BatchNormalization())
            model1.add(Conv1D(20, 4, padding='same',  activation='relu'))
            model1.add(BatchNormalization())
            model1.add(Conv1D(40, 4, padding='same',  activation='relu'))
            model1.add(BatchNormalization())
            model1.add(MaxPooling1D(2,strides=2, padding='same'))
            model1.add(BatchNormalization())
            model1.add(Dense(128, activation='relu'))
            model1.add(BatchNormalization())
            ##model1.add(Dropout(0.25))


            if (not train_only_with_cir):
                model2 = Sequential()
                model2.add(tf.keras.layers.Input(shape=(inputSizeNoCir,),name='input_2'))
                model2.add(Reshape((1,inputSizeNoCir)))
                model2.add(Dense(64, activation='relu'))
                model2.add(BatchNormalization())
                ##model2.add(Dropout(0.25))

            # CONCATENATE INPUTS
            if (not train_only_with_cir):
                output = tf.keras.layers.Concatenate()([model1.output, model2.output])
                #output = Dense(16, activation='relu')(output)
                output = Dense(1, activation='sigmoid')(output)
            else:
                #output = Dense(16, activation='relu')(model1.output)
                output = Dense(1, activation='sigmoid')(model1.output)

            #output = Dense(1, activation='sigmoid')(output)

            # FINAL MODEL
            if (not train_only_with_cir):
                model = tf.keras.Model(inputs=[model1.input, model2.input], outputs=[output])
            else:
                model = tf.keras.Model(inputs=[model1.input], outputs=[output])
            
            tf.keras.utils.plot_model(model, "model_two_branches.png", show_shapes=True, dpi=300, show_layer_names=False)
            #tf.keras.utils.plot_model(model, "model.png")
            #print(model.summary())
            #print('### Network parameters count: ' + str(model.count_params()))

            ada_grad = tf.keras.optimizers.Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=ada_grad,
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=['accuracy',f1_m,precision_m, recall_m])

            start_time = time.time()
            history = model.fit(
                train_dataset_all,
                epochs=epochs,
                # validation_data= val_dataset_all,
                shuffle=False,
                verbose=1
            )
            executionTime = time.time() - start_time
            print(executionTime)
            resultsExecutioinTime.append(executionTime)

            # Test model
            [loss, accuracy, f1_score, precision, recall] = model.evaluate(test_dataset_all)
            resultsAccuracy.append(accuracy)
            resultsF1.append(f1_score)
            resultsPrecision.append(precision)
            resultsRecall.append(recall)

        print('++++++++++++++ END +++++++++++++++++')
        print('MODE: ' + modeStr[mode])
        print('Num Reps: ' + str(numReps))
        print('Accuracy: ' + str(resultsAccuracy))
        print('Time: ' + str(resultsExecutioinTime))
        print('......................................')

        if usesPdp:
            pdpLabel = '_pdp_'+ str(pdp_size)
        else:
            pdpLabel=''

        np.save('Results_'+version+'/execution_'+str(mode) + pdpLabel + cir_energy_mode_label, resultsExecutioinTime)
        np.save('Results_'+version+'/accuracy_'+str(mode) + pdpLabel+cir_energy_mode_label, resultsAccuracy)
        np.save('Results_'+version+'/f1_'+str(mode) +pdpLabel+cir_energy_mode_label, resultsF1)
        np.save('Results_'+version+'/precision_'+str(mode) +pdpLabel+cir_energy_mode_label, resultsPrecision)
        np.save('Results_'+version+'/recall_'+str(mode) + pdpLabel+cir_energy_mode_label, resultsRecall)

    # model.save('TrainedModels/losnlos_'+str(mode)) 
    #show_batch(raw_train_data)