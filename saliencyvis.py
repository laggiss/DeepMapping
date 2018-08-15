import os

from random import randint



from keras.applications.imagenet_utils import preprocess_input

from scipy import misc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import load_model

import keras.backend as K

import cv2

#from loader import loadAsScalars



K.set_learning_phase(False)

"""

Double Check all datas to see if something is wrong during process data augmentation

"""

IMG_SIZE = 224

INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)

IMG_DIR = "f:/test"
#Define the img size

IMG_SIZE  = 224

INPUT_DIM = (IMG_SIZE, IMG_SIZE, 3)



#Define directories

baseDir = r"D:\Arnaud\data_croutinet\ottawa\data"

trainDir = os.path.join(baseDir, "train/train.csv")

validationDir = os.path.join(baseDir, "validation/validation.csv")

testDir = os.path.join(baseDir, "test/test.csv")

roads_loubna_dir = os.path.join(baseDir, "roads_loubna")

ranking_dir = os.path.join(baseDir, "rankingNoSigmoid")

activation_dir = os.path.join(ranking_dir, "activation")

check_dir = os.path.join(ranking_dir, "checkdata")

models_dir = os.path.join(baseDir, "models")

fake_dir = r'F:\test'#os.path.join(baseDir,"fake_dataset")
#destdir=fake_dir
vision_model=load_model("f:/models/finetunded_gt_956_percent_fc_and_block5_and_block4.h5")#load_model(os.path.join(models_dir, "scoreNetworkNoSigmoid.h5"))
model=vision_model.get_layer("vgg19")


dlist=["F:/ottawa_image_db/45.395289,-75.734107",
        "F:/ottawa_image_db/45.353038,-75.743813",
        "F:/ottawa_image_db/45.430856,-75.634009",
        "F:/ottawa_image_db/45.387784,-75.759421",
        "F:/ottawa_image_db/45.331311,-75.790179",
        "F:/ottawa_image_db/45.368548,-75.779295",
        "F:/ottawa_image_db/45.391854,-75.759665",
        "F:/ottawa_image_db/45.440927,-75.671258",
        "F:/ottawa_image_db/45.405719,-75.726039",
        "F:/ottawa_image_db/45.381848,-75.760526",
        "F:/ottawa_image_db/45.412198,-75.674896"]


def copyFiles(pathlist,dest_dir=fake_dir):
    import glob
    import shutil
    for s in pathlist:
        for filename in glob.glob(os.path.join(s, '*.*')):
            shutil.copy(filename, dest_dir)


def loadImageFix(name):

    """

    from an name, load the coresponding array of pixels

    :param name: name of the pictures

    :return: array of pixels, constiuting the picture

    """

    img  = misc.imread(os.path.join(IMG_DIR, name))

    img  = misc.imresize(img, (IMG_SIZE, IMG_SIZE))

    return img

def loadAsScalars(path):

    """

    load a

    :param path: the csv file were all your duels are saved

    Mine has the format :

    name of left image, name of right image, label

    :return: a tuple containing arrays of pictures of left, right images, their name, and labels

    """

    leftImages = []

    rightImages = []

    labels = []

    namesLeft = []

    namesRight = []

    with open(path, 'r') as csvfileReader:

        reader = csv.reader(csvfileReader, delimiter=',')

        for line in reader:

            if line != [] and line[2] != '0.5':

                leftImages.append(loadImageFix(line[0]))

                rightImages.append(loadImageFix(line[1]))

                labels.append(int(line[2]))

                namesLeft.append(line[0])

                namesRight.append(line[1])



    leftImages = np.array(leftImages)

    rightImages = np.array(rightImages)


    labels = np.array(labels)



    leftImages = preprocess_input(x=np.expand_dims(leftImages.astype(float), axis=0))[0]

    rightImages = preprocess_input(x=np.expand_dims(rightImages.astype(float), axis=0))[0]



    leftImages = leftImages.astype('float32')# / 255

    rightImages = rightImages.astype('float32')# / 255



    return (leftImages, rightImages, labels, namesLeft, namesRight)




def loadImage(name, dir = roads_loubna_dir):

    img  = misc.imread(os.path.join(dir, name))

    img  = misc.imresize(img, (IMG_SIZE, IMG_SIZE))

    return img





def heatmap(picture, picture_path):

    image = np.expand_dims(picture, axis=0)

    last_conv_layer = model.get_layer('block5_conv4')

    grads = K.gradients(model.get_output_at(0), last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))



    iterate = K.function([model.get_input_at(0)], [pooled_grads, last_conv_layer.output[0]])



    pooled_grads_value, conv_layer_output_value = iterate([image])



    for i in range(512):

        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)



    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)

    #plt.matshow(heatmap)



    img = cv2.imread(picture_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    return superimposed_img

def plotFakeDataSet():

    filenames = os.listdir(fake_dir)

    images = []

    heatmaps = []

    for file in filenames:

        images.append(loadImage(file, fake_dir))



    images = np.array(images)

    images_preprocess = preprocess_input(x=np.expand_dims(images.astype(float), axis=0))[0]

    #predictions = model.predict([images_preprocess])



    for i in range(len(filenames)):

        path = os.path.join(fake_dir, filenames[i])

        heatmaps.append(heatmap(images_preprocess[i], path))

        cv2.imwrite(os.path.join(fake_dir, "score" +  filenames[i]), heatmaps[i])



    # for i in range(len(images)):
    #
    #
    #
    #     plt.figure()
    #
    #     plt.subplot(1, 2, 1)
    #
    #     #plt.title("score : " + str(predictions[i]))
    #
    #     plt.imshow(images[i])
    #
    #
    #
    #     plt.suptitle(filenames[i], fontsize=16)
    #
    #
    #
    #     plt.subplot(1, 2 , 2)
    #
    #     plt.imshow(loadImage( "score" +  filenames[i],fake_dir))
    #
    #     plt.savefig(os.path.join(fake_dir,  "score" +  filenames[i]))