import cv2
import numpy as np

from . import FeaturesConfig

class SiftFeaturesConfig(FeaturesConfig):
    """
    Parametres configurant extraction des features
    avec OpenCV SIFT
    """
    def __init__(self, executor=None, chunk_size=150):
        super().__init__(executor, chunk_size)

        self.install_path = "dataset/SiftFeatures.hd5"
        self.nfeatures = 500

    def features_width(self):
        return 128

    def features_dtype(self):
        return np.float32

    def create_factory(self):
        return cv2.xfeatures2d.SIFT_create(self.nfeatures)
