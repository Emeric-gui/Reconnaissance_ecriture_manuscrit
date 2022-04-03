


class Traitement:
    def __init__(self, cheminImage):
        self.__cheminImage = cheminImage
        self.__image = None
        self.__imageThreshold = None


    def pretraitement(self):
        print("Fonction pour faire les prétraitements sur l'image")
        self.__realignement()
        self.__lissage()
        self.__binarisation()


    def __realignement(self):
        """
            Pour réaligner le document qui a été numérisé
            Peut être utile si l'image a été tourné de quelques degrés
        """

    def __lissage(self):
        """
        Lissage des bords et enlève les taches
        :return:
        """

    def __binarisation(self):
        """
            utilisation de threshold pour avoir une image en noir et blanc
        :return:
        """

    def __zonage(self):
        """
            Identification des colonnes, paragraphes, legendes, etc
        :return:
        """
    def __ligne_and_words_detection(self):
        """
            Detection de lignes et de mots
        :return:
        """