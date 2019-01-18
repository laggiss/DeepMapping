import os
import sys
from random import randint
import numpy as np
import pandas as pd
import keras
from keras import Model
from keras import backend as K, metrics
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import SGD
from scipy import misc
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.models import load_model
## Import classes from same folder (only necessary when run interactively)
sys.path.append(r'F:/OneDrive/DeepMapping')
from datagen_class_aug_test import myDataGeneratorAug
from dataStep import DataSplit
import tensorflow as tf
K.clear_session()

# Attempt to reduce memory errors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

def compute_accuracy(predictions, labels):
    """Function that determines the score to compute. Here it is the cohen kappa
    score.
    """
    return cohen_kappa_score(predictions, labels)

def auc_score(labels, predictions):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, predictions)

K.set_image_dim_ordering('tf')

# Variable corresponding to the images size in pixels. 224 is the usual value
IMG_SIZE = 224

d=DataSplit()

model_save = 'F:/DeepMappingData/new/model_FC_Block5_952.h5'#finetunded_gt_956_percent_fc_and_block5_and_block4.h5'
classification_model = load_model(model_save)


X_test = np.load(d.x_test_fname)
y_test = np.load(d.y_test_fname)

pred = (classification_model.predict([X_test[:, 0], X_test[:, 1]]) > 0.5).astype(int)
tr_acc = compute_accuracy(pred, y_test)
f1=f1_score(y_test,pred)
ev = classification_model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, batch_size=24)

print("MODEL EVAL: {}".format(ev))
print("PRED KAPPA: {}".format(tr_acc))
print("AUC SCORE: {}".format(auc_score(y_test, pred)))
print("F1 SCORE: {}".format(f1))