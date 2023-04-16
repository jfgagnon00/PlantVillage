import cv2

from .extraction import _extract
from .FeaturesConfig import FeaturesConfig
from .key_points import _list_to_cv_key_points


class ExtractionAdapter():
    """
    Utilitaire encapsulant l'extraction de features d'une image
    et l'affichage des key points y correspondants
    """
    def __init__(self, config):
        self._desc_factory = config.create_factory()

    def extract(self, image):
        """
        Extraction des key points et features d'une image

        image:
            image a trairer

        retour:
            tuple (key points, features). Il est possible que le tuple
            soit (None, None) si aucun features n'est trouve
        """
        return _extract(self._desc_factory, image)

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
