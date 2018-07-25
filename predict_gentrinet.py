from __future__ import absolute_import, print_function

import sys
import glob
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from scipy import misc

sys.path.append(r"f:\onedrive\a11PyCharmCNN")
import tensorflow as tf
K.clear_session()

# Forces the keras images fomat to have the number of channels last
K.set_image_dim_ordering('tf')

# Variable corresponding to the images size in pixels. 224 is the usual value
IMG_SIZE = 224
model_save = "f:/deepmappingdata/exclusiontrain/exclusionmod.h5"#'F:/DeepMappingData/new/vgg19_siamese_base_train.h5'

with tf.device('/gpu:0'):

    classification_model = load_model(model_save)

    # Load previously trained model

    # We fetch all the folder paths in the folder containing the images in an array

    #locations = glob.glob('f:/ottawa_image_db/*')
    locations=[]
    cnt=0

    with open("f:/models/fout.csv",'r') as f:
        x=f.readlines()

    for l in x:
        locations.append('f:/ottawa_image_db\\'+l.rstrip("\n"))


    num_locations = len(locations)

    print('loading locations')

    # We proceed the classification by steps of 1/64 of the dataset so we don't run out of memory

    for portion in range(34):

        print(portion / 34)

        # X is the array containing the data to classify

        X = []

        # Array of the corresponding location and year

        loc_year = []

        # We calculate the indices bounds of the current portion

        lower_bound = int(np.ceil(num_locations * portion / 34))

        upper_bound = int(np.ceil(num_locations * (portion + 1) / 34))

        for loc_i in range(lower_bound, upper_bound):

            # print((loc_i-lower_bound)/(upper_bound-lower_bound))

            # we fetch the folder path of the current location

            loc = locations[loc_i]

            # We store all the image paths of this folder in an array

            images = glob.glob(loc + '/*.jpg')

            # If there is more the 1 image at this location

            if len(images) > 1:

                # latlon is the part of the name of the file that corresponds to the position

                # i.e. the folder name

                latlon = loc[-20:]

                index = 0

                # We go through the folder and add each couple to the dataset

                while index < (len(images) - 1):

                    path1 = images[index]

                    path2 = images[index + 1]

                    # The year corresponds with the first 4 characters in the file name

                    year1 = path1.split('\\')[-1][:4]

                    year2 = path2.split('\\')[-1][:4]

                    img1 = misc.imread(path1)[:610, :]

                    img2 = misc.imread(path2)[:610, :]

                    img1 = misc.imresize(img1, (IMG_SIZE, IMG_SIZE))

                    img2 = misc.imresize(img2, (IMG_SIZE, IMG_SIZE))

                    img1 = preprocess_input(x=np.expand_dims(img1.astype(float), axis=0))[0]

                    img2 = preprocess_input(x=np.expand_dims(img2.astype(float), axis=0))[0]

                    X.append([img1, img2])

                    loc_year.append([latlon, year1, year2])

                    index += 1

        X = np.array(X)


        # Predicting the data class: if the confindence that there is a change is more than 50% (0.5)

        # We assume that ther is a change, and in other cases we assume there is not

        pred = (classification_model.predict([X[:, 0], X[:, 1]]) > 0.5).astype(int)

        loc_year = np.array(loc_year)

        # Saving the results of the portion as a text file

        concatenation = np.concatenate((loc_year, pred), axis=1)

        np.save('f:/preds/predval/classification_results_all_images_%i.h5' % portion, concatenation)

    # test_array_X = 'c:/gist/testarrayX.npy'
    # test_array_Y = 'c:/gist/testarrayY.npy'
    # X_test = np.load(test_array_X)
    # y_test = np.load(test_array_Y)
    # ev = classification_model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, batch_size=24)

## reconsrruct
results = glob.glob('f:/preds/predval' + '/*.npy')
with open("f:/models/fullres_replicate_exclude.csv", 'w') as f:
    f.write('LAT,LONG,YEAR1,YEAR2,PRED\n')
    for item in results:
        print("item x")
        narray = np.load(item)
        for row in narray:
            stringout=','.join(map(str, row))
            f.write("{}\n".format(stringout))

with open("f:/models/fullres_replicate_onesonly.csv", "w") as o:
    with open("f:/models/fullres_replicate_exclude.csv", 'r') as f:
        f.__next__()
        o.write('LAT,LONG,YEAR1,YEAR2,PRED\n')
        for i in f:
            lst=i.split(",")
            if lst[4][0]=='1':
                o.write(i)

