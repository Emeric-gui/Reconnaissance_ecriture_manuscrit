# on import les modules pour créer l'interface
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
# on importe la librairie pour enregistrer dans un word
import docx

import os
from traitement import Traitement


class Fenetre:
    def __init__(self):
        self.__filename = None
        self.__window = tk.Tk()
        self.__window.geometry("500x500")  # Size of the window
        self.__window.title('Scribo')
        my_font1 = ('times', 14, 'bold')
        l1 = tk.Label(self.__window, text='Selectioner votre image', width=30, font=my_font1)
        l1.grid(row=1, column=1)  # on met notre texte en ligne 1 colonne 1

        l2 = tk.Label(self.__window, text='Selectioner votre image', width=30, font=my_font1)
        l2.grid(row=2, column=1)  # on met notre texte en ligne 1 colonne 1
        b1 = tk.Button(self.__window, text='Upload Files', width=20, command=lambda: self.__upload_file())
        b1.grid(row=3, column=1)  # on met le bouton en dessous du texte soit en ligne 2 colonne 2

    def __upload_file(self):
        file_types = [('Images', '*.jpg;*.png;*.jpeg'),
                      ('All', '*.*')]  # type of files to select
        self.__filename = tk.filedialog.askopenfilename(title="Choisir une image", filetypes=file_types)

        # on va ensuite afficher l'images choisie
        img = Image.open(self.__filename)
        img = img.resize((200, 200))  # new width & height
        img = ImageTk.PhotoImage(img)
        e1 = Label(image=img)
        e1.image = img
        e1.grid(row=4, column=1)

        b2 = tk.Button(self.__window, text='Transformer en Word', width=20, command=lambda: self.__imgToTxt())
        b2.grid(row=5, column=1)

    def __imgToTxt(self):
        # on apply ici la fonction de traitement de l'image
        self.__applyTraitement(self.__filename)

        mydoc = docx.Document()  # on ouvre le word / le crée s'il existe pas
        mydoc.add_paragraph("test")  # on écrit dedan (a remplacer par les vrais lettres
        wordname = "NewDoc.docx"
        path = "/"
        mydoc.save(wordname)
        e2 = Label(self.__window, text=wordname)
        e2.grid(row=6, column=1)

    def activateWindow(self):
        self.__window.mainloop()  # Keep the window open

    def __applyTraitement(self, image):

            print("Image : {0}".format(image))
            cheminImage = image

            traitement = Traitement(cheminImage)
            traitement.pretraitement()
            # traitement.show_image()
