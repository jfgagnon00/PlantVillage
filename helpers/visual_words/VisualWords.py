import cv2

from ..features.extraction import _extract as _features_extract
from ..features.key_points import _list_to_cv_key_points


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

        TODO: a reviser?
    """
    def __init__(self, features_config, bovw_model):
        self._desc_factory = features_config.create_factory()
        self._bovw_model = bovw_model

    def extract(self, image):
        """
        Extraction des key points, features et visual words
        contenu dans une image

        image:
            image a trairer

        retour:
            tuple (key_points, features, visual_words). Il est possible que le tuple
            soit (None, None, None)
        """
        key_points, features = _features_extract(self._desc_factory, image)
        if features is None:
            return None, None, None

        return key_points, features, None

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
        return cv2.drawKeypoints(image,
                                 _list_to_cv_key_points(key_points),
                                 None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
