import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
from tensorflow_datasets.core.visualization import show_examples
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import tensorflow.python.ops.numpy_ops.np_config as np_config


class Classifier:

    def __init__(self):
        np_config.enable_numpy_behavior()
        self.__num_classes = 37
        self.__input_shape = (28, 28, 1)
        self.__model = None

        self.__is_juste_letters = True

        # a = 97 ascii A = 65

        self.__dict_label_all = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",

            10: "A",
            11: "B",
            12: "C",
            13: "D",
            14: "E",
            15: "F",
            16: "G",
            17: "H",
            18: "I",
            19: "J",
            20: "K",
            21: "L",
            22: "M",
            23: "N",
            24: "O",
            25: "P",
            26: "Q",
            27: "R",
            28: "S",
            29: "T",
            30: "U",
            31: "V",
            32: "W",
            33: "X",
            34: "Y",
            35: "Z",

            36: "a",
            37: "b",
            38: "c",
            39: "d",
            40: "e",
            41: "f",
            42: "g",
            43: "h",
            44: "i",
            45: "j",
            46: "k",
            47: "l",
            48: "m",
            49: "n",
            50: "o",
            51: "p",
            52: "q",
            53: "r",
            54: "s",
            55: "t",
            56: "u",
            57: "v",
            58: "w",
            59: "x",
            60: "y",
            61: "z"
        }
        self.__dict_label_letters = {
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "E",
            6: "F",
            7: "G",
            8: "H",
            9: "I",
            10: "J",
            11: "K",
            12: "L",
            13: "M",
            14: "N",
            15: "O",
            16: "P",
            17: "Q",
            18: "R",
            19: "S",
            20: "T",
            21: "U",
            22: "V",
            23: "W",
            24: "X",
            25: "Y",
            26: "Z",
        }
        name_model = "archi_lettre22.h5"
        if not os.path.exists(name_model):
            (self.__x_train, self.__y_train), (self.__x_test, self.__y_test) = emnist.load_data(
                    type='letters')
            # self.__plot_images()
            # self.__align_images()
            self.makeCNN()
        else:
            self.load_model(name_model)

    def __plot_images(self):
        fig, ax = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True)
        ax = ax.flatten()
        for k in range(1, 27):
            print("k : "+str(k))
            img = self.__x_train[self.__y_train == k][0].reshape(28, 28)
            ax[k].imshow(img, cmap='gray', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        # for image in self.__x_train:
            # cv2.imshow("window", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def __scales_images(self):
        # Scale images to the [0, 1] range
        self.__x_train = self.__x_train.astype("float32") / 255
        self.__x_test = self.__x_test.astype("float32") / 255
        # Make sure images size is (28, 28, 1)
        self.__x_train = np.expand_dims(self.__x_train, -1)
        self.__x_test = np.expand_dims(self.__x_test, -1)
        print("x_train shape:", self.__x_train.shape)
        print(self.__x_train.shape[0], "train samples")
        print(self.__x_test.shape[0], "test samples")

    def __convert_classes(self):
        # Convert class vectors to binary class matrices
        self.__y_train = keras.utils.to_categorical(self.__y_train, self.__num_classes)
        self.__y_test = keras.utils.to_categorical(self.__y_test, self.__num_classes)
        self.__model = keras.Sequential(
            [
                keras.Input(shape=self.__input_shape),
                layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.5),
                layers.Conv2D(128, kernel_size=(3, 3), padding="valid", activation="relu"),
                layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.__num_classes, activation="softmax"),
            ]
        )

    def __set_model(self):
        pass

    def __classifier_summary(self):
        self.__model.summary()
        # plot_model(self.__model)

    def __fit_generator(self):
        batch_size = 256
        epochs = 20

        self.__model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.__model.fit(self.__x_train, self.__y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def __predict_generator(self):
        score = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def __save_model(self):
        """Save the model in order to use it later"""
        self.__model.save('archi_lettre22.h5')

    def makeCNN(self):
        """Only for training"""
        self.__scales_images()
        self.__convert_classes()
        self.__set_model()
        self.__classifier_summary()
        self.__fit_generator()
        self.__save_model()
        self.__predict_generator()


    def load_model(self, name_model):
        """Apply this function if the model is already train"""
        self.__model = load_model(name_model)


    def __resize_image(self, image):
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, -1)
        print("image shape:", image.shape)
        # Scale images to the [0, 1] range
        image = image.astype("float32") / 255
        # Make sure images size is (28, 28, 1)
        image = tf.image.resize(image, [28, 28])
        print("image shape after resize:", image.shape)
        return image

    def reality(self, image, make_capi=False):
        """Va servir de fonction pour les prédictions"""
        image = self.__resize_image(image)
        pred = self.__model.predict(image)
        pred_max = np.argmax(pred[0], axis=-1)
        pourcent_max = np.amax(pred[0], axis=-1)*100

        pred_delete_max = np.delete(pred[0], pred_max)
        pred_max_2 = np.argmax(pred_delete_max, axis=-1)
        pourcent_max_2 = np.amax(pred_delete_max, axis=-1)*100

        pred_delete_max_2 = np.delete(pred_delete_max, pred_max_2)
        pred_max_3 = np.argmax(np.delete(pred_delete_max_2, pred_max_2), axis=-1)
        pourcent_max_3 = np.amax(pred_delete_max_2, axis=-1)*100

        pred_delete_max_3 = np.delete(pred_delete_max_2, pred_max_3)
        pred_max_4 = np.argmax(np.delete(pred_delete_max_3, pred_max_3), axis=-1)
        pourcent_max_4 = np.amax(pred_delete_max_3, axis=-1)*100

        pred_delete_max_4 = np.delete(pred_delete_max_3, pred_max_4)
        pred_max_5 = np.argmax(np.delete(pred_delete_max_4, pred_max_4), axis=-1)
        pourcent_max_5 = np.amax(pred_delete_max_4, axis=-1)*100

        val_retour = None

        if not self.__is_juste_letters:

            if not make_capi:
                if pred_max in range(10, 36):
                    pred_max = pred_max + 26
            else:
                if pred_max in range(36, 63):
                    pred_max = pred_max - 26
            val_retour = self.__dict_label_all.get(pred_max)

        else:
            lettre = self.__dict_label_letters.get(pred_max)

            if not make_capi:
                if 65 <= ord(lettre) <= 90:
                    val_lettre = ord(lettre) + 32
                    val_retour = chr(val_lettre)
                else:
                    val_retour = lettre

            else:
                if 97 <= ord(lettre) <= 122:
                    val_lettre = ord(lettre) - 32
                    val_retour = chr(val_lettre)
                else:
                    val_retour = lettre



        # val_retour_2 = self.__dict_label_all.get(pred_max_2)
        # val_retour_3 = self.__dict_label_all.get(pred_max_3)
        # val_retour_4 = self.__dict_label_all.get(pred_max_4)
        # val_retour_5 = self.__dict_label_all.get(pred_max_5)
        print("retour 1 : {0} | {1} %".format(val_retour, pourcent_max))
        # print("retour 2 : {0} | {1} %".format(val_retour_2, pourcent_max_2))
        # print("retour 3 : {0} | {1} %".format(val_retour_3, pourcent_max_3))
        # print("retour 4 : {0} | {1} %".format(val_retour_4, pourcent_max_4))
        # print("retour 5 : {0} | {1} %".format(val_retour_5, pourcent_max_5))

        return val_retour
