"""
TODO: refactor avec Sift
"""
import cv2
import h5py
import numpy as np
import os
import uuid

from ..Concurrent import parallel_for
from ..Jupyter import display_html
from ..MetaObject import MetaObject
from tqdm.notebook import tqdm


_FEATURES_KEY = "features"
_FEATURES_WIDTH = 32
_FEATURES_DTYPE = np.uint8

_KEYPOINTS_KEY = "keypoints"
_KEYPOINTS_WIDTH = 7
_KEYPOINTS_DTYPE = np.float32

_INDICES_KEY = "indices"
_INDICES_DTYPE = np.uint16


def _cv_key_points_to_list(cv_key_points):
    converted = []
    for cv_kp in cv_key_points:
        kp = [cv_kp.pt[0], cv_kp.pt[1],
              cv_kp.size,
              cv_kp.angle,
              cv_kp.response,
              float(cv_kp.octave),
              float(cv_kp.class_id)]
        converted.append(kp)
    return converted

def _list_to_cv_key_points(key_points):
    converted = []
    for kp in key_points:
        cv_kp = cv2.KeyPoint(kp[0], kp[1], # pt
                             kp[2], # size
                             kp[3], # angle
                             kp[4], # response
                             int(kp[5]),  # octave
                             int(kp[6]))  # class_id
        converted.append(cv_kp)
    return converted

def _batch_extract(config, h5_file, batch_iterables):
    features = np.empty((0, _FEATURES_WIDTH), dtype=_FEATURES_DTYPE)
    key_points = np.empty((0, _KEYPOINTS_WIDTH), dtype=_KEYPOINTS_DTYPE)
    indices = np.empty((0,), dtype=_INDICES_DTYPE)

    desc_factory = cv2.ORB_create(config.nfeatures)

    count = 0
    for index, zip_info, image_reader in batch_iterables:
        count += 1

        if zip_info.is_dir():
            continue

        image = image_reader.get_image(zip_info)
        kpts, descs = desc_factory.detectAndCompute(image, None)
        if descs is None:
            continue

        features = np.append(features,
                            descs,
                            axis=0)

        key_points = np.append(key_points,
                               _cv_key_points_to_list(kpts),
                               axis=0)

        indices = np.append(indices,
                            [index] * descs.shape[0],
                            axis=0)

    if features.shape[0] > 0:
        batch_name = str(uuid.uuid4())
        h5_file.create_dataset(batch_name,
                               data=features)

        batch_key_points = batch_name + "-key_points"
        h5_file.create_dataset(batch_key_points,
                               data=key_points)

        batch_indices = batch_name + "-index"
        h5_file.create_dataset(batch_indices,
                               data=indices)

        batch = MetaObject.from_kwargs(
                features_count=features.shape[0],
                features_ds_name=batch_name,
                key_points_ds_name=batch_key_points,
                indices_ds_name=batch_indices)
    else:
        batch = None

    return count, batch

def _batch_extract_threaded(config, h5_file, orb_features_iter):
    batch_accum = MetaObject.from_kwargs(
        features_count=0,
        features_ds_names=[],
        key_points_ds_names=[],
        indices_ds_names=[])

    with tqdm(total=orb_features_iter.count) as progress:
        def _batch_done(batch_results):
            count, batch = batch_results

            progress.update(count)

            if not batch is None:
                batch_accum.features_count += batch.features_count
                batch_accum.features_ds_names.append(batch.features_ds_name)
                batch_accum.key_points_ds_names.append(batch.key_points_ds_name)
                batch_accum.indices_ds_names.append(batch.indices_ds_name)

        parallel_for(orb_features_iter,
                     _batch_extract,
                     config,
                     h5_file,
                     task_completed=_batch_done,
                     executor=config.executor,
                     chunk_size=config.chunk_size)

    return batch_accum

def _batch_merge(h5_file, batch_accum):
    features_layout = h5py.VirtualLayout((batch_accum.features_count, _FEATURES_WIDTH),
                                         dtype=_FEATURES_DTYPE)

    key_points_layout = h5py.VirtualLayout((batch_accum.features_count, _KEYPOINTS_WIDTH),
                                           dtype=_KEYPOINTS_DTYPE)

    indices_layout = h5py.VirtualLayout((batch_accum.features_count,),
                                        dtype=_INDICES_DTYPE)

    start = 0
    for feature_name, \
        key_point_name, \
        indices_name in zip(batch_accum.features_ds_names,
                            batch_accum.key_points_ds_names,
                            batch_accum.indices_ds_names):
        features = h5_file[feature_name]
        key_points = h5_file[key_point_name]
        indices = h5_file[indices_name]

        stop = start + features.shape[0]

        features_layout[start:stop, ...] = h5py.VirtualSource(features)
        key_points_layout[start:stop, ...] = h5py.VirtualSource(key_points)
        indices_layout[start:stop] = h5py.VirtualSource(indices)

        start = stop

    h5_file.create_virtual_dataset(_FEATURES_KEY, features_layout)
    h5_file.create_virtual_dataset(_KEYPOINTS_KEY, key_points_layout)
    h5_file.create_virtual_dataset(_INDICES_KEY, indices_layout)
    h5_file.flush()

class OrbFeaturesConfig():
    """
    Parametres configurant extraction des features
    avec OpenCV ORB
    """
    def __init__(self, executor=None, chunk_size=150):
        self.install_path = "dataset/OrbFeatures.hd5"
        self.force_generate = False
        self.read_only = True
        self.nfeatures = 500
        self.executor = executor
        self.chunk_size = chunk_size

class OrbFeaturesIter():
    """
    Pas la meilleur interface; a reviser
    """
    def __init__(self,
                 zip_file,
                 iterable_index_imagepath,
                 iterable_count=-1):
        self._zip_file = zip_file
        self._iterable_index_imagepath = iterable_index_imagepath
        self._iterable_count = iterable_count

    @property
    def count(self):
        return self._iterable_count

    @property
    def zip_file(self):
        return self._zip_file

    def get_image(self, zip_info):
        return None

    def __iter__(self):
        return self

    def __next__(self):
        index, image_path = next(self._iterable_index_imagepath)
        return index, self._zip_file.getinfo(image_path), self

def orb_features_load(config, orb_features_iter):
    """
    Utilitaire encapsulant extraction avec OpenCV ORB.

    config:
        Instance de OrbFeaturesConfig

    orb_features_iter:
        Instance de OrbFeaturesIter

    Retour:
        MetaObject encapsulant les features. Si le fichier demande existe,
        il est retourne sinon il est construit a partir de orb_features_iter.
    """
    def instantiate():
        mode = "r" if config.read_only else "r+"

        h5_file = h5py.File(config.install_path, mode)
        h5_features = h5_file[_FEATURES_KEY]
        h5_key_points = h5_file[_KEYPOINTS_KEY]
        h5_indices = h5_file[_INDICES_KEY]

        return MetaObject.from_kwargs(h5_file=h5_file,
                                      features=h5_features,
                                      key_points=h5_key_points,
                                      indices=h5_indices)

    if not config.force_generate and \
       os.path.exists(config.install_path):
        return instantiate()

    if config.force_generate or \
       not os.path.exists(config.install_path):
        dataset_path, _ = os.path.split(config.install_path)
        os.makedirs(dataset_path, exist_ok=True)

    display_html("<b>Extraire ORB features</b>")
    with h5py.File(config.install_path, "w") as h5_file:
        batch_accum = _batch_extract_threaded(config,
                                              h5_file,
                                              orb_features_iter)
        _batch_merge(h5_file, batch_accum)

    return instantiate()

def orb_features_draw_key_points(orb_features, orb_features_iter):
    """
    Utilitaire permettant d'afficher les key points

    config:
        Instance de OrbFeaturesConfig

    orb_features:
        Object creee par orb_features_load()

    Retour:
        Image contenant les key points.
    """

    indices = orb_features.indices[...]

    for index, zip_info, image_reader in orb_features_iter:
        image = image_reader.get_image(zip_info)
        where = np.where(indices == index)
        key_points = orb_features.key_points[where][...]
        key_points = _list_to_cv_key_points(key_points)

        yield index, \
              zip_info.filename, \
              len(key_points), \
              cv2.drawKeypoints(image,
                                key_points,
                                None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
