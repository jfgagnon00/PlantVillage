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

        figure = Figure(figsize=(6, 4), dpi=100)
        self._axes = figure.subplot_mosaic("ABB;CDD")
        self._image_ax = self._axes["A"]
        self._image_kpts_ax = self._axes["C"]
        self._tf_idf_ax = self._axes["B"]
        self._info_ax = self._axes["D"]

        file_select = tk.Button(self, text ="Choisir image", command=self._select_file)
        file_select.pack()

        self._canvas = FigureCanvasTkAgg(figure, self)
        self._clear_ui()
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.title("Bag of Visual Words Demo")

    def _select_file(self):
        filename = tk.filedialog.askopenfilename()
        if not filename is None and len(filename) > 0:
            self._refresh(filename)

    def _refresh(self, filename):
        # filename = "dataset/demo/Grape___healthy/image (1).JPG"
        # filename = "dataset/demo/Grape___Esca_(Black_Measles)/image (45).JPG"
        # filename = "dataset/demo/astronaut.jpg"

        image = cv2.imread(filename)

        if image is None:
            print("Erreur lecture", filename)
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        key_points, features, tf_idf, label = self._visual_words.predict(image)
        image_kpts = self._visual_words.draw_key_points(image, key_points)

        self._clear_ui()
        self._update_graph(image, image_kpts, tf_idf)
        self._update_infos(filename, key_points, features, tf_idf, label)
        self._canvas.draw()

    def _clear_ui(self):
        self._image_ax.clear()
        self._image_ax.set_title("Image")
        self._image_ax.set_xticks([])
        self._image_ax.set_yticks([])

        self._image_kpts_ax.clear()
        self._image_kpts_ax.set_title("Image key points")
        self._image_kpts_ax.set_xticks([])
        self._image_kpts_ax.set_yticks([])

        self._tf_idf_ax.clear()
        self._tf_idf_ax.set_title("Image tf-idf")

        self._info_ax.clear()
        self._info_ax.set_title("Info")
        self._info_ax.set_xticks([])
        self._info_ax.set_yticks([])

    def _update_graph(self, image, image_kpts, tf_idf):
        self._image_ax.imshow(image)
        self._image_kpts_ax.imshow(image_kpts)
        self._tf_idf_ax.bar(range(tf_idf.shape[1]), np.ravel(tf_idf))

    def _update_infos(self, filename, key_points, features, tf_idf, label):
        x = 0.01
        y = 0.85
        dy = 0.1

        self._info_ax.text(x, y - 0*dy, filename)
        self._info_ax.text(x, y - 1*dy, f"# key points: {len(key_points)}")
        self._info_ax.text(x, y - 2*dy, f"Descripteur shape: {features.shape}")
        self._info_ax.text(x, y - 3*dy, f"TF-IDF shape: {tf_idf.shape}")
        self._info_ax.text(x, y - 6*dy, f"Classe: {label}", color="red", fontsize="large")


if __name__ == "__main__":
    # relire le modele pre-entraine
    print("Chargement model")
    configs = hlp.get_configs("config_overrides.json")
    visual_words = VisualWords.from_sift_configs(configs)

    print("Initialisation application")
    app = VisualWordsApp(visual_words)
    app.geometry("1200x600")
    app.mainloop()
