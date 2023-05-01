import cv2
import helpers as hlp
import matplotlib
import numpy as np
import tkinter as tk

matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from helpers.visual_words.VisualWords import VisualWords


class VisualWordsApp(tk.Tk):
    def __init__(self, visual_words):
        super().__init__()

        self._visual_words = visual_words

        # obtenir une image
        # self._image = cv2.imread("dataset/demo/Grape___healthy/image (1).JPG")
        # self._image = cv2.imread("dataset/demo/Grape___Esca_(Black_Measles)/image (45).JPG")
        self._image = cv2.imread("dataset/demo/astronaut.jpg")

        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        self._image = cv2.resize(self._image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        key_points, features, tf_idf, label = self._visual_words.predict(self._image)

        self._image_kpts = self._visual_words.draw_key_points(self._image, key_points)

        self.title("Bag of Visual Words Demo")

        figure = Figure(figsize=(6, 4), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self)

        axes = figure.subplot_mosaic("AB;CC")
        axes["A"].imshow(self._image)
        axes["A"].set_xticks([])
        axes["A"].set_yticks([])

        axes["B"].imshow(self._image_kpts)
        axes["B"].set_xticks([])
        axes["B"].set_yticks([])

        axes["C"].bar(range(tf_idf.shape[1]), np.ravel(tf_idf))

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        file_select = tk.Button(self, text ="Hello", command=self._select_file)
        file_select.pack()

    def _select_file(self):
        print("Select File")

    def _refresh(self):
        pass

    def _display_stats(self):
        #     # afficher les resultats
        #     print("Image shape:", image.shape)
        #     print("Key points:", len(key_points))
        #     print("Features shape:", features.shape)
        #     print("TF-IDF shape:", tf_idf.shape)
        #     print("Label:", label)
        pass


if __name__ == "__main__":
    # relire le modele pre-entraine
    print("Loading model")
    configs = hlp.get_configs("config_overrides.json")
    visual_words = VisualWords.from_sift_configs(configs)

    print("Starting app")
    app = VisualWordsApp(visual_words)
    app.geometry("1200x600")
    app.mainloop()
