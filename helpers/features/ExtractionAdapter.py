from .extraction import _extract
from .FeaturesConfig import FeaturesConfig


class ExtractionAdapter():
    """
    Utilitaire encapsulant l'extraction de features d'une image.
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
