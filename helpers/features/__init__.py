import cv2
import h5py
import numpy as np
import os


from .extraction import _batch_extract_parallel, \
                        _update_train_test, \
                        _FEATURES_KEY, \
                        _INDICES_PREFIX, \
                        _KEYPOINTS_KEY, \
                        _TRAIN_FEATURES_KEY, \
                        _TEST_FEATURES_KEY
from .key_points import _list_to_cv_key_points
from .DatasetIter import DatasetIter
from .FeaturesConfig import FeaturesConfig
from .OrbFeaturesConfig import OrbFeaturesConfig
from .SiftFeaturesConfig import SiftFeaturesConfig

from ..MetaObject import MetaObject


def _update_train_test_properties(features):
    h5_train_features = None
    if _TRAIN_FEATURES_KEY in features.h5_file:
        h5_train_features = features.h5_file[_TRAIN_FEATURES_KEY]

    h5_test_features = None
    if _TEST_FEATURES_KEY in features.h5_file:
        h5_test_features = features.h5_file[_TEST_FEATURES_KEY]

    return MetaObject.override_from_kwargs(features,
                                           train_features=h5_train_features,
                                           test_features=h5_test_features)

def _instantiate(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)
    h5_features = h5_file[_FEATURES_KEY]
    h5_key_points = h5_file[_KEYPOINTS_KEY]
    h5_index_features = h5_file[f"{_INDICES_PREFIX}/{_FEATURES_KEY}"]
    h5_index_key_points = h5_file[f"{_INDICES_PREFIX}/{_KEYPOINTS_KEY}"]

    features = MetaObject.from_kwargs(h5_file=h5_file,
                                      features=h5_features,
                                      key_points=h5_key_points,
                                      index_to_features=h5_index_features,
                                      index_to_key_points=h5_index_key_points)

    _update_train_test_properties(features)

    return features

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

def update_train_test(features, train_indices, test_indices):
    """
    Met a jour (ou creee) les datasets train/test des features

    features:
        MetaObject obtenu par load

    train_indices, test_indices:
        Index des images a utiliser pour populer les datasets
    """
    _update_train_test(features.h5_file, train_indices, test_indices)
    _update_train_test_properties(features)

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
    for index, image_path, image_future in dataset_iter:
        image = image_future()
        key_points = features.index_to_key_points[str(index)][...]
        key_points = _list_to_cv_key_points(key_points)

        yield index, \
              image_path, \
              len(key_points), \
              cv2.drawKeypoints(image,
                                key_points,
                                None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
