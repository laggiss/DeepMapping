"""
This script was modified from gentriNetConvServer2 for retraining of network to ensure:
 1. No information leakage of the validation set to the training weight adjustments;
 2. Include data augmentation in the training set;
 3. vgg image normalization of training and validation data separately using keras imagenet preprocesssing
 4. Fine tuning of the vgg bottleneck layers (conv-layers).
"""
# Importing the keras framework
from __future__ import absolute_import, print_function
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.models import load_model
## Import classes from same folder (only necessary when run interactively)
sys.path.append(r'F:/OneDrive/DeepMapping')#F:/OneDrive/DeepMapping)
from datagen_class_aug_test import myDataGeneratorAug
from dataStep import DataSplit

K.clear_session()


## Create instance of DataSplit class to get paths
#
d=DataSplit()
#
## Create instance of DataSplit class and run create method to create new datasets
##   and return paths with instance of class
# d.create()


# Constants
input_path=d.output_path

history_continue_training = input_path+os.sep+'mdict14try_vgg_norm.npy'  # "c:/gist/mdict14try.npy"
model_save = input_path+os.sep+'vgg19_siamese_base_train.h5'
weights_file = input_path+os.sep+'vgg19_siamese_base_train_weights.h5'
#check_point_weights = "c:/gist/gentrinet_checkpoint_weights.h5"


# fix random seed for reproducibility
# np.random.seed(7)

# Forces the keras images fomat to have the number of channels last
K.set_image_dim_ordering('tf')

# Variable corresponding to the images size in pixels. 224 is the usual value
IMG_SIZE = 224

valid_results = np.load(d.validation_results_save)
train_results = np.load(d.training_results_save)  # c:/gist/all_results.npy')
X_valid = np.load(d.x_valid_fname)
y_valid = np.load(d.y_valid_fname)
X_test = np.load(d.x_test_fname)
y_test = np.load(d.y_test_fname)
yes=pd.read_pickle(d.yes_fname)
no=pd.read_pickle(d.no_fname)


def compute_accuracy(predictions, labels):
    """Function that determines the score to compute. Here it is the cohen kappa
    score.
    """
    return cohen_kappa_score(predictions, labels)

def auc_score(labels, predictions):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, predictions)

def the_model(run_new=False, unfreeze=False):
    vision_model = VGG19(include_top=False,
                         weights='imagenet',
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Definition of the 2 inputs
    img_a = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    img_b = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Outputs of the vision model corresponding to the inputs
    # Note that this method applies the 'tied' weights between the branches
    out_a = vision_model(img_a)
    out_b = vision_model(img_b)

    # Concatenation of these outputs
    concat = keras.layers.concatenate([out_a, out_b])

    # For fine tuning make convolution base not-trainable with unfreeze=False (bottleneck feature extractor)
    # Make trainable to fine tune top convolution layers
    if unfreeze:
        vision_model.trainable = True
        print("Vision model is trainable")
    else:
        vision_model.trainable = False
        print("Vision model is not trainable")

    # Definition of the top layers
    concat = Flatten()(concat)
    concat = Dropout(0.3)(concat)
    concat = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name="Dense1")(concat)
    concat = Dropout(0.3)(concat)
    concat = Dense(4096, activation='relu', kernel_initializer='glorot_normal', name="Dense2")(concat)
    concat = Dropout(0.1)(concat)
    concat = Dense(1000, activation='relu', kernel_initializer='glorot_normal', name="Dense3")(concat)
    main_out = Dense(1, activation='sigmoid')(concat)

    # The classification model is the full model: it takes 2 images as input and
    # returns a number between 0 and 1. The closest the number is to 1, the more confident
    # it is that there was a change
    classification_model = Model([img_a, img_b], main_out)

    if not run_new:
        classification_model.load_weights("f:/models/sw.h5")  # ""f:/models/backup_92_fc_only.h5")#
        # "f:/models/vgg19_siamese_14th_try_vgg_norm_fine_tuned_29_Dec_17_514pm.h5")#weights_file)#,by_name=True))#
        print("Weights loaded..")

    # Unfreeze top convolution layers for second step of fine tuning after training model tuning only top FC layers
    if unfreeze:
        set_trainable = False
        for layer in classification_model.get_layer('vgg19').layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        print("Block_5 Convolution unfrozen")

    # Definition of the SGD optimizer, that allows to fine tune learning rate (very important)
    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.5, nesterov=True)
    # Compilation of the model using a loss used for binary problems
    classification_model.compile(loss='binary_crossentropy', optimizer=sgd,
                                 metrics=[metrics.binary_accuracy, metrics.mae, metrics.mse])

    return classification_model


# # With run_new = False, load all_results from
# run_new = False
# # Training/Validation data split...
# if run_new:
#  #moved to createDataSplit()
# else:
    # load list in format of


run_new=True
if run_new:
    # save all history params
    print("\n\nNew run of model\n\n")
    mdict = {
        'binary_accuracy': [], 'loss': [], 'mean_absolute_error': [], 'mean_squared_error': [],
        'val_binary_accuracy': [], 'val_loss': [], 'val_mean_absolute_error': [], 'val_mean_squared_error': []
        }
    classification_model = the_model(run_new=True)
else:
    print("\n\nLoading existing model\n\n")
    mdict = np.load(history_continue_training)
    mdict = mdict.tolist()
    classification_model = load_model(
        model_save)  # finetunded_gt_939_percent_fc_and_block5_class_weights.h5")

# number of wanted iterations. Each iteration takes a different set of augmented images so
# keep this number high (depending on the learning rate) when training the model
n_iter = 8
kappas = []
test_eval = []
# Initialisation of a confusion matrix
conf_mat = np.zeros((2, 2))
threashold = 0.85
for iteration in range(n_iter):
    print(iteration / n_iter)

    # sample positive and negative cases for current iteration. It is faster to use fit on batch of n yesno and augment
    # that batch using datagen_class_aug_test than to use fit_generator with the datagen_class_aug_test and small batch
    # sizes.
    yesno = yes.sample(700).append(no.sample(3500))
    labels = dict(zip([str(x) for x in yesno.index.tolist()],
                      [1 if x == '1' else 0 for x in yesno.winner.tolist()]))
    partition = {'train': list(zip([str(x) for x in yesno.index.tolist()], zip(yesno.left_id, yesno.right_id)))}

    batchSizeAug = len(yesno.index.tolist())
    # Set-up variables for augmentation of current batch of yesno in partition
    params = {
        'dim_x': 224,
        'dim_y': 224,
        'dim_z': 3,
        'batch_size': batchSizeAug,
        'shuffle': True
        }

    datagenargs = {
        'rotation_range': 2, 'width_shift_range': 0.2, 'height_shift_range': 0.2,
        'shear_range': 0.1,
        'zoom_range': 0.25, 'horizontal_flip': True, 'fill_mode': 'nearest'
        }

    training_generator = myDataGeneratorAug(**params).generate(labels, partition['train'], seed=randint(1, 10000),
                                                               datagenargs=datagenargs)

    X, y = training_generator.__next__()
    # zero center images
    X = np.array(X)
    # X=X.astype(float) - np.mean(X, axis=0)
    # Fitting the model.
    cls_wt = {0: 10, 1: 90}
    #checkpointer = ModelCheckpoint(filepath=check_point_weights, verbose=1, save_best_only=True)
    mout = classification_model.fit([X[0], X[1]],
                                    y,
                                    batch_size=24,
                                    epochs=50,
                                    validation_data=([X_valid[:, 0], X_valid[:, 1]], y_valid),
                                    verbose=1, class_weight=cls_wt)  # ,callbacks=[checkpointer])
    # mout = classification_model.fit_generator(generator=training_generator,
    #                                           steps_per_epoch=len(partition['train']) // params['batch_size'],
    #                                           epochs=1,
    #                                           validation_data=([X_valid[:, 0], X_valid[:, 1]], y_valid))  # ,reduce_lr])

    # Concatenate history
    for k, v in mout.history.items():
        for i in v:
            mdict[k].append(i)

    # Predictions to qulify the classification quality
    pred = (classification_model.predict([X_valid[:, 0], X_valid[:, 1]]) > 0.5).astype(int)
    tr_acc = compute_accuracy(pred, y_test)
    ev = classification_model.evaluate([X_valid[:, 0], X_valid[:, 1]], y_valid, batch_size=24)
    test_eval.append(ev)
    print("MODEL EVAL: {}".format(ev))
    print("PRED KAPPA: {}".format(tr_acc))
    print("AUC SCORE: {}".format(auc_score(y_test, pred)))
    # pred = (classification_model.predict([X_test[:, 0], X_test[:, 1]]) > 0.5).astype(int)
    # te_acc = compute_accuracy(pred, y_test)
    kappas.append(tr_acc)
    conf_mat = confusion_matrix(y_test, pred)
    print(conf_mat)

    if ev[1] > threashold:
        print("\n\nSaving model....\n\n")
        classification_model.save(model_save)
        threashold = ev[1]
    #classification_model.save_weights(weights_file)  # )weights_file)#

# classification_model.save(model_save)
#classification_model.save("f:/models/finetunded932percent_fc_and_block5.h5")


np.save(history_continue_training, np.array(mdict))



classification_model = load_model(model_save)

pred = (classification_model.predict([X_valid[:, 0], X_valid[:, 1]]) > 0.5).astype(int)
tr_acc = compute_accuracy(pred, y_valid)
ev = classification_model.evaluate([X_valid[:, 0], X_valid[:, 1]], y_valid, batch_size=24)

print("MODEL EVAL: {}".format(ev))
print("PRED KAPPA: {}".format(tr_acc))
print("AUC SCORE: {}".format(auc_score(y_test, pred)))

import matplotlib.pyplot as plt

plt.figure(41, figsize=(12, 8), dpi=150)
loss = mdict['loss']
val_loss = mdict['val_loss']
epochs = range(1, len(loss) + 1)
plt.subplot(211)
plt.plot(epochs, loss, 'b', label='Training loss', color='r')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
acc = mdict['binary_accuracy']
val_acc = mdict['val_binary_accuracy']
plt.plot(epochs, acc, 'b', label='Training acc', color='r')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

kappas = np.array(kappas)
print('')
print("Kappa: %0.10f (+/- %0.2f)" % (kappas.mean(), kappas.std()))
print('')
print(conf_mat)

x = {'acc': [1], 'loss': [1], 'val_acc': [1], 'val_loss': [1]}

mdict = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
for k, v in x.items():
    for i in v:
        mdict[k].append(i)

import collections

super_dict = collections.defaultdict(list)

for k, v in mout.history.items():  # d.items() in Python 3+
    super_dict[k].add(v)

super_dict = {}

for k, v in mout.history.items():  # d.items() in Python 3+
    super_dict.setdefault(k, []).append(v)
# These first 3 lines save the model as a json file
# model_json = classification_model.to_json()
# with open('vgg19_siamese_4th_ver.json', 'w') as jsonfile:
#	jsonfile.write(model_json)

# This last line is very important: it saves the weights of the model you trained for
# so long (warning: the weights files live up to their name and are very big, up to 1GB with this model)
# Warning: don't forget to change the name at each iteration
# classification_model.save_weights('vgg19_siamese_13th_try.h5')

import matplotlib.pyplot as plt

ss = -80
plt.figure(41)
loss = mdict['loss'][ss:][:1000]
val_loss = mdict['val_loss'][ss:][:1000]
epochs = range(1, len(loss) + 1)
plt.subplot(211)
plt.plot(epochs, loss, 'b', label='Training loss', color='r')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
acc = mdict['acc'][ss:][:1000]
val_acc = mdict['val_acc'][ss:][:1000]
plt.plot(epochs, acc, 'b', label='Training acc', color='r')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from keras.models import Model, load_model

oldmodel = load_model("f:/models/finetunded_gt_956_percent_fc_and_block5_and_block4.h5")

# classification_model = the_model(True,False)
#
# #Some sanity checking with assertions.
# assert len(oldmodel.layers) == len(classification_model.layers)
# for idx,layer in enumerate(oldmodel.layers):
#     assert type(oldmodel.layers[idx]) == type(classification_model.layers[idx])
#     classification_model.layers[idx].set_weights(layer.get_weights())
#
# sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.5, nesterov=True)
# # Compilation of the model using a loss used for binary problems
# classification_model.compile(loss='binary_crossentropy', optimizer=sgd,
#                                  metrics=[metrics.binary_accuracy, metrics.mae, metrics.mse])
import numpy as np

X_test = np.load(test_array_X)
y_test = np.load(test_array_Y)
ev = oldmodel.evaluate([X_test[:, 0], X_test[:, 1]], y_test, batch_size=24)

## Visu
layer_dict = dict([(layer.name, layer) for layer in oldmodel.layers])
from keras import backend as K

layer_name = 'block5_conv3'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = oldmodel.get_layer('vgg19').get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, X_test[0, :, :, :])[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([X_test[0, :, :, :]], [loss, grads])

