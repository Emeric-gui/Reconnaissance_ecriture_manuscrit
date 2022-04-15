import cv2
import os
import sys
from traitement import Traitement

"""On commence par récupérer le repertoire d'images

    Il contiendra les différentes sous images

    Dans les arguments du programme entrez le chemin relatif du dossier où sont stockées les images.

"""

args = sys.argv
repertoireStockImages = args[1]


from interface import Fenetre

fenetre = Fenetre()
fenetre.activateWindow()

#
# # Repertoire qui contient toutes les images
# liste_images = os.listdir(repertoireStockImages)
# i = 1
#
# # Parcours de toutes les images et traitement sur chacune d'elles
# for image in liste_images:
#     print("Image : {0}".format(image))
#     cheminImage = repertoireStockImages + "/" + image
#
#     traitement = Traitement(cheminImage)
#     traitement.pretraitement()
#     traitement.show_image()
