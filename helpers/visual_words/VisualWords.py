import numpy as np
import pickle

from . import load_bovw
from .processing import _extract as _vw_extract
from ..features import draw_key_points as _features_draw_key_points
from ..features.extraction import _extract as _features_extract


class VisualWords():
    """
    Utilitaire encapsulant l'extraction de visual words
    a partir d'une image.

    NOTE:
        L'interface est un peu differente des autres
        modules. Les pixels de l'image sont passes plutot
        qu'un index dans le dataset. L'idee etant qu'on
        prepare un outil qui va servir pour une demo.
        L'usager pourra selectioner une image de son choix.

        Les autres modules sont plus oriente en traitement
        par lot. Ici, on traite 1 element a la fois.
    """
    def __init__(self):
        self._desc_factory = None
        self._bovw_model = None
        self._bovw_idf = None
        self._n_clusters = -1
        self._classifier_model = None

    @classmethod
    def from_sift_configs(cls, configs):
        """
        Factory methode
        """
        sift_bovw = load_bovw(configs.sift_bovw, None)

        instance = VisualWords()
        instance._desc_factory = configs.sift.create_factory()
        instance._bovw_model = sift_bovw.model
        instance._bovw_idf = sift_bovw.idf
        instance._n_clusters = sift_bovw.cluster_centers.shape[0]

        with open(configs.sift_classifier.install_path, "rb") as file:
            instance._classifier_model = pickle.load(file)

        return instance

    def predict(self, image):
        """
        Extraction des key points, features, visual words et label
        contenu dans une image (traitement 1 image a la fois)

        image:
            image a trairer

        retour:
            tuple (key_points, features, tf-idf, label). Il est possible que le tuple
            soit (None, None, None, None)
        """
        key_points, features_array = _features_extract(self._desc_factory, image)
        if features_array is None:
            return None, None, None, None

        visual_words_freq = _vw_extract(self._bovw_model, self._n_clusters, features_array)
        tf_idf = np.multiply(visual_words_freq, self._bovw_idf)
        label = self._classifier_model.predict(tf_idf)

        return key_points, \
               features_array, \
               tf_idf, \
               label

    def draw_key_points(self, image, key_points):
        """
        Utilitaire pour obtenir une image avec ses keypoints.

        image:
            image qui a servi a obtenir key_points et features

        key_points:
            key points obtenus par extract()

        Retour:
            image contenant les key points
        """
        return _features_draw_key_points(image, key_points)
