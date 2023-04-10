import cv2
import h5py
import numpy as np
import os

from .extraction import _batch_extract_parallel, \
                        _FEATURES_KEY, \
                        _INDICES_KEY, \
                        _KEYPOINTS_KEY
from .key_points import _list_to_cv_key_points
from .DatasetIter import DatasetIter
from .FeaturesConfig import FeaturesConfig
from .Orb import OrbFeaturesConfig
from .Sift import SiftFeaturesConfig

from ..MetaObject import MetaObject


def _instantiate(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)
    h5_features = h5_file[_FEATURES_KEY]
    h5_key_points = h5_file[_KEYPOINTS_KEY]
    h5_indices = h5_file[_INDICES_KEY]

    return MetaObject.from_kwargs(h5_file=h5_file,
                                    features=h5_features,
                                    key_points=h5_key_points,
                                    indices=h5_indices)

def load(config, dataset_iter):
    """
    Utilitaire encapsulant extraction/loading features.

    config:
        Instance de FeaturesConfig

    dataset_iter:
        Instance de DatasetIter

    Retour:
        MetaObject encapsulant les features. Si le fichier demande existe,
        il est retourne sinon il est construit a partir de dataset_iter.
    """
    if not config.force_generate and os.path.exists(config.install_path):
        return _instantiate(config)

    if config.force_generate or not os.path.exists(config.install_path):
        path, file  = os.path.split(config.install_path)
        if not file is None:
            os.makedirs(path, exist_ok=True)

    print("Extraire features")
    with h5py.File(config.install_path, "w") as h5_file:
        _batch_extract_parallel(config,
                                h5_file,
                                dataset_iter)

    return _instantiate(config)

def key_points_iter(features, dataset_iter):
    """
    Utilitaire pour obtenir une image avec ses keypoints.
    Fonctionne comme un generateur

    features:
        Object creee par load_features()

    dataset_iter:
        Instance de DatasetIter

    Retour:
        Voir ci-bas
    """
    indices = features.indices[...]

    for index, image_path, image_future in dataset_iter:
        image = image_future()
        where = np.where(indices == index)
        key_points = features.key_points[where][...]
        key_points = _list_to_cv_key_points(key_points)

        yield index, \
              image_path, \
              len(key_points), \
              cv2.drawKeypoints(image,
                                key_points,
                                None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
