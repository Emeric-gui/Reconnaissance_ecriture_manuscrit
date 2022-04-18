import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist


class Classifier:

    def __init__(self):
        (self.__input_train, self.__target_train), (self.__input_test, self.__target_test) = emnist.load_data(
            type='byclass')
        self.__num_classes = 62
        self.__input_shape = (28, 28, 1)
        (self.__x_train, self.__y_train), (self.__x_test, self.__y_test) = keras.datasets.fashion_mnist.load_data()

        self.__model = None

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

    def __set_model(self):
        self.__model = keras.Sequential(
            [
                keras.Input(shape=self.__input_shape),
                layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                # layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                # layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                # layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.__num_classes, activation="softmax"),
            ]
        )

    def __classifier_summary(self):
        self.__model.summary()

    def __fit_generator(self):
        batch_size = 128
        epochs = 15

        self.__model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.__model.fit(self.__x_train, self.__y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    def __predict_generator(self):
        score = self.__model.evaluate(self.__x_test, self.__y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def __save_model(self):
        """Save the model in order to use it later"""
        self.__model.save('archi_finale.h5')

    def makeCNN(self):
        self.__scales_images()
        self.__convert_classes()
        self.__set_model()
        self.__classifier_summary()
        self.__fit_generator()
        self.__save_model()
        self.__predict_generator()


classifier = Classifier()
classifier.makeCNN()
