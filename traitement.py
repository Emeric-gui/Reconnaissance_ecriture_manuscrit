import os

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plot


class Traitement:
    def __init__(self, cheminImage):
        """
            Constructeur
        :param cheminImage: chemin vers l'image
        """
        self.__directory_sub_images = "sub_images/"
        self.__directory_words = "words/"
        self.__directory_letters = "letters/"

        if not os.path.exists("sub_images"):
            os.mkdir("sub_images")

        if not os.path.exists("words"):
            os.mkdir("words")

        if not os.path.exists("letters"):
            os.mkdir("letters")

        self.__cheminImage = cheminImage
        self.__name_image = None

        self.__image_couleur = None
        self.__image_gris = None
        self.__image_threshold = None
        self.__image_contour = None

        self.__name_image = "image"

        self.__image_couleur = cv2.imread(self.__cheminImage)
        self.__image_couleur = self.__changeSizeImages(325, 250, self.__image_couleur)

    def pretraitement(self):
        """
            Application des traitements sur l'image de base, puis application sur les sous-images
        """
        # Image
        self.__image_gris = cv2.cvtColor(self.__image_couleur, cv2.COLOR_BGR2GRAY)
        self.__image_threshold = self.__binarisation(self.__image_gris)
        imageRetour = self.__zonage(self.__image_threshold)
        self.__image_contour = self.__lines_detection(imageRetour, self.__image_couleur.copy(),
                                                      self.__name_image,
                                                      self.__directory_sub_images)

        # Sous-images
        self.__plot_images('sub_images')
        sub_images = os.listdir(self.__directory_sub_images)
        for sub_image_tile in sub_images:
            self.__traitement_sub(sub_image_tile)
            os.remove(self.__directory_sub_images + sub_image_tile)

        # Afficher toutes les lettres dans un plot
        """Vu qu'on a trouvé toutes les lettres, on va afficher le tout dans un plot 
            pour pouvoir les supprimer du dossier
        """
        self.__plot_images('letters')

    # ------------------------------------------------------------------------------------

    def __binarisation(self, image):
        """
            utilisation de threshold pour avoir une image en noir et blanc
        :return: l'image thresholdée
        """
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    def __zonage(self, image):
        """
            Forme les rectangle autour des pixels blancs
        :param image: image thresholdée
        :return: retourne les rectangles autour des textes
        """
        # use morphology dilate to blur horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (175, 10))
        morph = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        # use morphology open to remove thin lines from dotted lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        return morph

    def __lines_detection(self, img_retour, img_couleur, nom_image, name_directory):
        """
            Detection des lignes
        :param img_retour:
        :param img_couleur:
        :param nom_image:
        :param name_directory:
        :return: image couleur avec les contours
        """
        w, h = 0, 0
        # find contours
        cntrs = cv2.findContours(img_retour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntrs = cntrs[::-1]
        # Draw contours excluding the topmost box
        result = img_couleur.copy()
        increment = 1
        for c in cntrs:
            box = cv2.boundingRect(c)
            x, y, w, h = box
            x2 = x + w
            y2 = y + h

            # create a sub image for each word in a temp directory
            """ After this we will go through each image to find each letter of each word"""
            sub_image = img_couleur.copy()[y:y2, x:x2, ::]
            if increment < 10:
                cv2.imwrite(name_directory + nom_image + "_0" + str(increment) + ".jpg", sub_image)
            else:
                cv2.imwrite(name_directory + nom_image + "_" + str(increment) + ".jpg", sub_image)

            cv2.rectangle(result, (x, y), (x2, y2), (0, 0, 255), 2)
            increment += 1

        return result

    # ------------------------------------------------------------------------------------

    def __zonage_sub(self, image):
        """
            Forme les rectangles autour des pixels blancs
        :param image: image thresholdée
        :return: retourne les rectangles autour des textes
        """
        # use morphology erode to blur horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 10))
        morph = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        return morph

    def __words_detection(self, img_retour, img_couleur, nom_image, name_directory):
        """
            Detection des mots
        :param img_retour:
        :param img_couleur:
        :param nom_image:
        :param name_directory: nom du répertoire où seront stockées les images des mots
        :return:
        """
        # find contours
        cntrs = cv2.findContours(img_retour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntrs = self.__stabilize_contours(cntrs)

        result = img_couleur.copy()

        increment = 1
        for c in cntrs:
            box = cv2.boundingRect(c)
            x, y, w, h = box
            x2 = x + w
            y2 = y + h + 5

            if w * h > 250:
                # create a sub image for each word in a temp directory
                """ After this we will go through each image to find each letter of each word
                """
                sub_image = img_couleur.copy()[y:y2, x:x2, ::]
                if increment < 10:
                    cv2.imwrite(name_directory + nom_image + "_0" + str(increment) + ".jpg", sub_image)
                else:
                    cv2.imwrite(name_directory + nom_image + "_" + str(increment) + ".jpg", sub_image)
                cv2.rectangle(result, (x, y), (x2, y2), (0, 0, 255), 2)
                increment += 1

        return result

    def __stabilize_contours(self, contours):
        """
           Pour trier l'ordre des mots dans la ligne, pour aller de gauche a droite
        :param contours: Contours trouvés dans l'image
        :return:
        """
        # Tri a bulles pour trier les valeurs selon le x
        pos = 0
        contours = list(contours)
        cntrs_length = len(contours)
        for i in range(0, cntrs_length):
            for j in range(0, cntrs_length - i - 1):
                if contours[j][pos][pos][pos] > contours[j + 1][pos][pos][pos]:
                    temp = contours[j]
                    contours[j] = contours[j + 1]
                    contours[j + 1] = temp
        return contours

    def __traitement_sub(self, sub_image_title):
        """
            Traite les différentes sous-images, c'est à dire la détection de mots à partir des lignes trouvées
            précédemment
        :param sub_image_title: nom de l'image
        :return:
        """
        name_sub_image = sub_image_title.split(".")[0]
        sub_image = cv2.imread(self.__directory_sub_images + sub_image_title)

        sub_image_gris = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        sub_image_threshold = self.__binarisation(sub_image_gris)

        sub_image_retour = self.__zonage_sub(sub_image_threshold)

        self.__words_detection(sub_image_retour, sub_image, name_sub_image,
                               self.__directory_words)

        # Après avoir trouvés les mots, on va chercher les caractères
        self.__plot_images('words')
        word_images = os.listdir(self.__directory_words)
        for word_image_tile in word_images:
            self.__traitement_word(word_image_tile)
            os.remove(self.__directory_words + word_image_tile)

    # ------------------------------------------------------------------------------------
    def __zonage_caracter(self, image):
        """
            Forme les rectangle autour des pixels blancs
        :param image: image thresholdée
        :return: retourne les rectangles autour des textes
        """
        # use morphology dilation to blur vertically
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 12))
        morph = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

        # erosion pour réduire la taille des élements
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel_erode)

        return morph

    def __caracter_detection(self, img_retour, img_couleur, nom_image, name_directory):
        """
            Detection de lignes et de mots
        :param img_retour:
        :param img_couleur:
        :param words:
        :return:
        """
        # find contours
        cntrs = cv2.findContours(img_retour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        cntrs = self.__stabilize_contours(cntrs)
        result = img_couleur.copy()

        img_gray = cv2.cvtColor(img_couleur.copy(), cv2.COLOR_BGR2GRAY)
        img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        increment = 1
        for c in cntrs:
            box = cv2.boundingRect(c)
            x, y, w, h = box
            x2 = x + w
            y2 = y + h + 5

            # create a sub image for each word in a temp directory
            """ After this we will go through each image to find each letter of each word"""

            # apply a threshold to obtain only black and white color on the image
            sub_image = img_threshold[y:y2, x:x2]
            sub_image = self.__add_border_image(sub_image)
            sub_image = self.__changeSizeImages(28, 28, sub_image)
            if increment < 10:
                cv2.imwrite(name_directory + nom_image + "_0" + str(increment) + ".jpg", sub_image)
            else:
                cv2.imwrite(name_directory + nom_image + "_" + str(increment) + ".jpg", sub_image)
            cv2.rectangle(result, (x - 2, y), (x2, y2), (0, 0, 255), 2)
            increment += 1

        return result

    def __add_border_image(self, image):
        """
            Ajoute des bordures blanches en plus de l'image
        :param image:
        :return: retourne l'image
        """
        black = [0, 0, 0]
        replicate = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=black)
        return replicate

    def __traitement_word(self, word_image_title):
        """
        :param sub_image_title:
        :return:
        """
        name_word_image = word_image_title.split(".")[0]

        word_image = cv2.imread(self.__directory_words + word_image_title)
        word_image_gris = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)
        word_image_threshold = self.__binarisation(word_image_gris)

        word_image_retour = self.__zonage_caracter(word_image_threshold)

        self.__caracter_detection(word_image_retour, word_image, name_word_image,
                                  self.__directory_letters)

    # ------------------------------------------------------------------------------------

    def __changeSizeImages(self, newSizeH, newSizeL, image):
        """
            Changer la taille d'une image tout en gardant un maximum de clarté par rapport à l'image originale
        :param newSizeH: nouvelle hauteur
        :param newSizeL: nouvelle longueur
        :param image: image à modifier
        :return: la nouvelle image avec sa nouvelle taille
        """
        return cv2.resize(image, (newSizeL, newSizeH), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------------------------
    def __plot_images(self, repertoire):
        """
            Afficher les différentes images contenues dans un répertoire
        :param repertoire: chemin vers le répertoire
        :return:
        """
        list_images = os.listdir(repertoire)
        nombre_images = len(list_images)
        nb_colonnes = 5
        nb_lignes = int(nombre_images / nb_colonnes)
        if nombre_images % nb_colonnes > 0:
            nb_lignes += 1

        fig, ax = plt.subplots(nrows=nb_lignes, ncols=nb_colonnes, sharex=True, sharey=True)
        ax = ax.flatten()

        i = 0
        for picture in list_images:
            img = cv2.imread(repertoire + "/" + picture)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            i += 1
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    def __print_image_opencv(self, image):
        """
            Affichage de l'image grâce à OpenCV
        :param image: image finale à afficher après segmentation
        :return:
        """

        cv2.imshow("fenetre_image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __print_image_matplot(self, image):
        """
            Affichage de l'image en utilisant des graphiques
            grace a la bibliotheque matplotlib
        :param image: image finale à afficher après segmentation
        :return:
        """
        fig, ax = plot.subplots(figsize=(20, 10))

        if len(image.shape) == 2:
            plot.imshow(image, cmap='gray')
        else:
            plot.imshow(image)
        plot.axis('off')
        ax.set_title(self.__name_image)
        plot.show()

    def __show_with_matplotlib(self, color_img, title, max_size, pos, gray=False):
        """Afficher une image avec matplotlib"""

        ax = plot.subplot(1, max_size, pos)

        if gray:
            img_RGB = color_img
            plot.imshow(img_RGB, cmap="gray")
        else:
            # On convertit l'image BGR en RGB
            img_RGB = color_img[:, :, ::-1]
            plot.imshow(img_RGB)

        plot.title(title)
        plot.axis('on')

    def __print_multiple_images_matplot(self, *images):
        nb_images = len(images)
        indice = 1
        plot.subplots(figsize=(20, 10), num=1)

        for image in images:
            isGray = False
            if len(image.shape) == 2:
                isGray = True
            self.__show_with_matplotlib(image, self.__name_image, nb_images, indice, gray=isGray)
            indice += 1
        plot.show()

    def __show_histogram_image(self, image):
        """
        Pour afficher l'histogramme d'intensité des pixels de l'image
        :param image: image qu'on souhaite étudier
        :return: None
        """
        n, bins, patches = plot.hist(image.flatten(), bins=range(256))
        plot.title(self.__name_image)
        plot.show()

    def show_image(self, image1, image2):
        """
            Appel des différentes fonctions selon la méthode d'affichage choisie
        :param image1:
        :param image2:
        :return:
        """
        self.__print_image_opencv(image1)
        self.__print_multiple_images_matplot(image2, image1)
        return None
    # ------------------------------------------------------------------------------------
