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

_INDICES_KEY = "indices"
_INDICES_DTYPE = np.uint16


def _batch_extract(config, h5_file, batch):
    features = np.empty((0, _FEATURES_WIDTH), dtype=_FEATURES_DTYPE)
    indices = np.empty((0,), dtype=_INDICES_DTYPE)

    desc_factory = cv2.ORB_create(config.nfeatures)

    count = 0
    for index, zip_info in batch:
        count += 1

        if zip_info.is_dir():
            continue

        # unzip in memory
        image = config.zip_file.read(zip_info)

        # create image from data
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        key_points, descs = desc_factory.detectAndCompute(image, None)
        if descs is None:
            continue

        features = np.append(features,
                            descs,
                            axis=0)

        indices = np.append(indices,
                            [index] * descs.shape[0],
                            axis=0)

    if features.shape[0] > 0:
        batch_name = str(uuid.uuid4())
        h5_file.create_dataset(batch_name,
                               data=features)

        batch_indices = batch_name + "-index"
        h5_file.create_dataset(batch_indices,
                               data=indices)

        batch = MetaObject.from_kwargs(
                features_count=features.shape[0],
                features_ds_name=batch_name,
                indices_ds_name=batch_indices)
    else:
        batch = None

    return count, batch

def _batch_extract_threaded(config, h5_file):
    batch_accum = MetaObject.from_kwargs(
        features_count=0,
        features_ds_names=[],
        indices_ds_names=[])

    with tqdm(total=config.iterable_count) as progress:
        def _batch_done(batch_results):
            count, batch = batch_results
            progress.update(count)

            if not batch is None:
                batch_accum.features_count += batch.features_count
                batch_accum.features_ds_names.append(batch.features_ds_name)
                batch_accum.indices_ds_names.append(batch.indices_ds_name)

        def _batch_generator():
            for index, image_path in config.iterable_index_imagepath:
                yield index, config.zip_file.getinfo(image_path)

        parallel_for(_batch_generator(),
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

    indices_layout = h5py.VirtualLayout((batch_accum.features_count,),
                                        dtype=_INDICES_DTYPE)

    start = 0
    for feature_name, indices_name in zip(batch_accum.features_ds_names, batch_accum.indices_ds_names):
        features = h5_file[feature_name]
        indices = h5_file[indices_name]

        stop = start + features.shape[0]

        features_layout[start:stop, ...] = h5py.VirtualSource(features)
        indices_layout[start:stop] = h5py.VirtualSource(indices)

        start = stop

    h5_file.create_virtual_dataset(_FEATURES_KEY, features_layout)
    h5_file.create_virtual_dataset(_INDICES_KEY, indices_layout)

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

        # not the best interface: better performance for the time being
        self.iterable_index_imagepath = None # chaque iter est un tuple (int, string)
        self.iterable_count = 0
        self.zip_file = None
        self.executor = executor
        self.chunk_size = chunk_size

def orb_features_load(config):
    """
    Utilitaire encapsulant preprocessing
    extraction avec OpenCV ORB

    config:
        Instance of OrbFeaturesConfig

    Retour:
        MetaObject encapsulant les features ou
        None si probleme.
    """
    def instantiate():
        mode = "r" if config.read_only else "r+"

        h5_file = h5py.File(config.install_path, mode)
        h5_features = h5_file[_FEATURES_KEY]
        h5_indices = h5_file[_INDICES_KEY]

        return MetaObject.from_kwargs(h5_file=h5_file,
                                      features=h5_features,
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
        batch_accum = _batch_extract_threaded(config, h5_file)
        _batch_merge(h5_file, batch_accum)

    return instantiate()
