import customtkinter
import threading
from tkinterdnd2 import TkinterDnD, DND_ALL
from tkinter import filedialog, ttk

import os


class CTkDnD(customtkinter.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class App(CTkDnD):
    def __init__(self):
        super().__init__()

        self.title("Detecteur Valeur Aberrante")
        self.geometry("500x500")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.dragdrop_frame = DragAndDropFrame(self)
        self.dragdrop_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(20, 10), sticky="nsew")

class DragAndDropFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.filenames = []  

        self.label = customtkinter.CTkLabel(self, text="Déposez ou cliquez pour choisir des fichiers", fg_color="gray30", corner_radius=6)
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.drop_target_register(DND_ALL)
        self.dnd_bind("<<Drop>>", self.on_drop)
        self.label.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Fichiers texte", "*.xlsx *.csv"), ("Tous les fichiers", "*.*")]
        )
        if file_paths:
            self.set_filenames(file_paths)

    def on_drop(self, event):
        file_paths = self.tk.splitlist(event.data)
        self.set_filenames(file_paths)

    def set_filenames(self, file_paths):
        self.filenames = list(file_paths)
        filenames_short = [os.path.basename(f) for f in self.filenames]
        label_text = f"{len(self.filenames)} fichier(s) sélectionné(s):\n" + ", ".join(filenames_short)
        self.label.configure(text=label_text)

    def get(self):
        return self.filenames


app = App()
app.mainloop()