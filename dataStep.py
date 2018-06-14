from __future__ import absolute_import, print_function

import os
import sys
import numpy as np
import pandas as pd
from keras.applications.vgg19 import preprocess_input
from scipy import misc


class DataSplit(object):

    def __init__(self,train_list_path= 'f:/deepmappingdata/retrain.txt',output_path='f:/deepmappingdata/new/',img_size=224):
        self.train_list_path = train_list_path
        self.output_path=output_path
        self.yes_fname=output_path+os.sep+"yes.npy"
        self.no_fname=output_path+os.sep+"no.npy"
        self.x_test_fname=output_path+os.sep+"testarrayX.npy"
        self.y_test_fname=output_path+os.sep+"testarrayY.npy"
        self.x_valid_fname=output_path+os.sep+"validarrayX.npy"
        self.y_valid_fname=output_path+os.sep+"validarrayY.npy"
        self.training_results_save=output_path+os.sep+"trainresultlist.npy"
        self.validation_results_save=output_path+os.sep+"validplustestresultlist.npy"
        self.IMG_SIZE=img_size


    def create(self):
        # Training uses a data generator that accepts a list of left,right,winner but the
        # validation and test arrays remain constant so can be created beforehand.
        # Currently the validation data is stored in the valid_results_continue_training file and
        # the training data in the train_results_continue_training file. If run_new=False then
        # these files are loaded so that hyperparamater optimization can be preformed without
        # the new model runs seeing the validation data at any time.

        # Array that contains all the results of the training dataset
        # all_results = []

        # Load the list of all training data
        all_results = np.loadtxt(self.train_list_path, str)

        # Separate training and validation sets
        id_shuffle = np.arange(all_results.shape[0])
        np.random.shuffle(id_shuffle)
        all_results = all_results[id_shuffle.flatten().tolist(), :]
        train_sep = all_results.shape[0] - int(0.4 * all_results.shape[0])

        train_results, valid_results = all_results[:train_sep, ], all_results[train_sep:, ]

        # split the training list into postive and negative lists for the datagenerator
        duelsDF = pd.DataFrame(train_results, None, ['left_id', 'right_id', 'winner'])
        mask_yes = duelsDF['winner'] == '1'
        yes = duelsDF[mask_yes]
        mask_no = duelsDF['winner'] == '0'
        no = duelsDF[mask_no]

        print("Created yes/no lists....")
        # Load validation images into array
        X_valid = []
        y_valid = []

        # If image adjustments are made then reload = True so that the same adjustments are made here
        # as in the datagen_class_aug_test class, e.g., preprocessing of imagenet data.
        cnt=0
        for row in valid_results:
            self.__drawProgressBar(cnt/valid_results.shape[0])
            if (row[2] == '0') | (row[2] == '1'):
                # print(row)
                # The label is the last value of the row
                y_valid.append(int(row[-1]))
                path_1 = row[0]
                path_2 = row[1]
                # Read the images and resize them, cropping bottom logo out, applying preprocessing
                img_1 = misc.imread(path_1)[:610, :]
                img_2 = misc.imread(path_2)[:610, :]
                img_1 = misc.imresize(img_1, (self.IMG_SIZE, self.IMG_SIZE))
                img_2 = misc.imresize(img_2, (self.IMG_SIZE, self.IMG_SIZE))
                img_1 = preprocess_input(x=np.expand_dims(img_1.astype(float), axis=0))[0]
                img_2 = preprocess_input(x=np.expand_dims(img_2.astype(float), axis=0))[0]
                X_valid.append([img_1, img_2])
            cnt+=1
        X_valid = np.array(X_valid)
        y_valid = np.array(y_valid)
        valid_id_shuffle = np.arange(y_valid.shape[0])
        np.random.shuffle(valid_id_shuffle)
        mid_p = len(valid_id_shuffle) // 2

        print("\nSaving files....")
        #
        pd.to_pickle(yes,self.yes_fname)
        pd.to_pickle(no,self.no_fname)
        np.save(self.training_results_save, train_results)
        np.save(self.validation_results_save, valid_results)
        np.save(self.x_valid_fname, X_valid[:mid_p, :])
        np.save(self.y_valid_fname, y_valid[:mid_p])
        np.save(self.x_test_fname, X_valid[mid_p:, :])
        np.save(self.y_test_fname, y_valid[mid_p:])

        #return [yes, no, X_valid[:mid_p, :], y_valid[:mid_p]]

    def __drawProgressBar(self,percent, barLen=20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()

x=DataSplit().create()
