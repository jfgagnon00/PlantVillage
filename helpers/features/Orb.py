import cv2
import numpy as np

from . import FeaturesConfig

class OrbFeaturesConfig(FeaturesConfig):
    """
    Parametres configurant extraction des features
    avec OpenCV ORB
    """
    def __init__(self, executor=None, chunk_size=150):
        super().__init__(executor, chunk_size)

        self.install_path = "dataset/OrbFeatures.hd5"
        self.nfeatures = 500

    def features_width(self):
        return 32

    def features_dtype(self):
        return np.uint8

    def create_factory(self):
        return cv2.ORB_create(self.nfeatures)
