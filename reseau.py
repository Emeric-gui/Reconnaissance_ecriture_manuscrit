import os.path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist
from tensorflow.keras.models import load_model
import tensorflow.python.ops.numpy_ops.np_config as np_config


class Classifier:

    def __init__(self, is_letter):
        """
            Constructeur pour créer un Classificateur
        :param is_letter: savoir si on ne recherche que des lettres ou des lettres ET chiffres
        """
        np_config.enable_numpy_behavior()
        self.__num_classes = 37
        self.__input_shape = (28, 28, 1)
        self.__model = None

        self.__is_juste_letters = is_letter
        self.__name_model_letter = "cnn_lettre.h5"
        self.__name_model_digit_letter = "cnn_chiffre_lettre.h5"

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

        name_model = ""
        type_load = ""
        # Pour récupérer le bon dataset
        if self.__is_juste_letters:
            name_model = self.__name_model_letter
            type_load = "letters"
        else:
            name_model = self.__name_model_digit_letter
            type_load = "byclass"

        if not os.path.exists(name_model):
            (self.__x_train, self.__y_train), (self.__x_test, self.__y_test) = emnist.load_data(
                    type=type_load)
            # self.__plot_images()
            self.makeCNN()
        else:
            self.load_model(name_model)

    # ------------------------------------------------------------------------------------

    def __plot_images(self):
        """
            Permet d'afficher différentes images du dataset
            Idéal pour faire des tests
        :return:
        """
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

    # ------------------------------------------------------------------------------------


    def __scales_images(self):
        """
            Les données d'entrainement sont "modifiées" pour quelles possèdent toutes la même taille
            et qu'elles correspondent à celle défini dans le réseau de neurone
        :return:
        """
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
        """
            Conversion des classes
            Et ecriture du modèle
        :return:
        """
        # Convert class vectors to binary class matrices
        self.__y_train = keras.utils.to_categorical(self.__y_train, self.__num_classes)
        self.__y_test = keras.utils.to_categorical(self.__y_test, self.__num_classes)
        self.__model = keras.Sequential(
            [
                layers.InputLayer(input_shape=self.__input_shape),
                layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.__num_classes, activation="softmax"),
            ]
        )

    def __classifier_summary(self):
        """
            Affiche le résumé du modèle créé auparavant
        :return:
        """
        self.__model.summary()

    def __fit_generator(self, batch_size=128, epochs=20):
        """
            Pour entrainer notre modèle, choix du batch_size et du nombre d'epochs
        :param batch_size:
        :param epochs:
        :return:
        """
        self.__model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.__model.fit(self.__x_train, self.__y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def __predict_generator(self):
        """
            Prédiction sur l'ensemble de test, pour valider notre modèle
        :return:
        """
        score = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def __save_model(self, name_model='archi_finale_lettre.h5'):
        """On enregistre l'architecture du modèle pour qu'on puisse utiliser un modèle pré-entrainé
            pour les prédictions
        """
        self.__model.save(name_model)

    def makeCNN(self):
        """
            Fonction qui appelle les différentes fonctions nécessaires à la création d'un nouveau modèle
        :return:
        """
        self.__scales_images()
        self.__convert_classes()
        self.__classifier_summary()
        self.__fit_generator()
        self.__save_model()
        self.__predict_generator()

    # ------------------------------------------------------------------------------------

    def load_model(self, name_model):
        """
            Charge un modèle pré-entrainé dans le modèle de notre programme
        :param name_model: Le nom du fichier où est contenu le modèle
        :return:
        """
        self.__model = load_model(name_model)

    def __resize_image(self, image):
        """
            Permet de changer la taille de l'image pour qu'elle soit acceptée par le modèle
        :param image: "matrice" de l'image
        :return: image avec une taille adapté pour le modèle
        """
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
        """
            Appeler pour faire des prédictions sur les lettres obtenus par détection de caractères
        :param image: image dont il faut trouver le label
        :param make_capi: booléen pour savoir si la lettre (image) être au milieu de la phrase
        :return:
        """
        image = self.__resize_image(image)
        pred = self.__model.predict(image)
        pred_max = np.argmax(pred[0], axis=-1)
        pourcent_max = np.amax(pred[0], axis=-1)*100

        val_retour = None

        # En fonction du modèle lettres ou lettres + nombres, les labels sont différents
        if not self.__is_juste_letters: # s'il y a des lettres et des nombres, on joue sur les indices du dictionnaire

            if not make_capi:
                if pred_max in range(10, 36):
                    pred_max = pred_max + 26
            else:
                if pred_max in range(36, 63):
                    pred_max = pred_max - 26
            val_retour = self.__dict_label_all.get(pred_max)

        else: # ici on joue sur les indices ascii des caractères
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

        print("retour 1 : {0} | {1} %".format(val_retour, pourcent_max))
        return val_retour
