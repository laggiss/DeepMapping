import numpy as np
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from random import randint
import sys
class myDataGeneratorAug(object):
    """Data Generator for Siamese model inputs"""

    def __init__(self, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True):
        """Initialization"""
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs, seed, datagenargs):
        """Generates batches of samples"""
        # Infinite loop
        while 1:
            sys.stdout.write("hello\n")
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            # print(indexes)
            # Generate batches
            imax = int(len(indexes) / self.batch_size)

            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                # print(list_IDs_temp)
                # Generate data
                X1, X2, y = self.__data_generation(labels, list_IDs_temp, seed, datagenargs)
                #print(list_IDs_temp)
                X1 = preprocess_input(x=np.expand_dims(X1.astype(float), axis=0))[0]
                X2 = preprocess_input(x=np.expand_dims(X2.astype(float), axis=0))[0]

                yield ([X1, X2], y)

    def __get_exploration_order(self, list_IDs):
        """Generates order of exploration"""
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp, seed, datagenargs):
        """Generates data of batch_size samples"""  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, 2, self.dim_x, self.dim_y, self.dim_z))
        # y = np.empty(self.batch_size, dtype=int)

        X1, X2, y = [], [], []
        cnt=0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            print(len(list_IDs_temp))
            self.__drawProgressBar(cnt / len(list_IDs_temp))
            # Store volume
            seed = randint(1,50000)#.random.seed()
            left_datagen = ImageDataGenerator(**datagenargs)  # , augment=True, seed=seed)
            right_datagen = ImageDataGenerator(**datagenargs)  # , augment=True, seed=seed)
            #print(i)
            # print(ID[1][0])
            img1 = misc.imread(ID[1][0])[:610,:]
            x_left = img1.reshape((1,) + img1.shape)
            x_lefta = left_datagen.flow(x_left, batch_size=1, seed=seed).next()
            #print(x_lefta.shape)
            x_leftb = misc.imresize(x_lefta[0, :], (self.dim_x, self.dim_y))
            #x_leftb=preprocess_input(x=np.expand_dims(x_leftb.astype(float), axis=0))[0]
            X1.append(x_leftb)

            img2 = misc.imread(ID[1][1])[:610,:]
            x_right = img2.reshape((1,) + img2.shape)
            x_righta = right_datagen.flow(x_right, batch_size=1, seed=seed).next()
            x_rightb = misc.imresize(x_righta[0, :], (self.dim_x, self.dim_y))
            #x_rightb=preprocess_input(x=np.expand_dims(x_rightb.astype(float), axis=0))[0]
            X2.append(x_rightb)

            y.append(labels[ID[0]])
            cnt+=1
        # print("somethingnewert")
        return np.array(X1), np.array(X2), np.array(y)
        # return X, y  # sparsify(y)

    def sparsify(y):
        """Returns labels in binary NumPy array"""
        n_classes = 2  # Enter number of classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(y.shape[0])])

    def __drawProgressBar(self,percent, barLen=20):
        sys.stdout.write("\r")
        sys.stdout.write("Using data generator for batch...")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()