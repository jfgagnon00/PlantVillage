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

_INDICES_KEY = "indices"
_INDICES_DTYPE = np.uint16

_LOOKUP_INDICES_KEY = "lookup_indices"
_LOOKUP_IMAGE_PATH_KEY = "lookup_image_path"
_LOOKUP_ENCODING = "utf-8"

def _extract_features(config, h5_file, batch_iterable):
    features = np.empty((0, _FEATURES_WIDTH), dtype=_FEATURES_DTYPE)
    indices = np.empty((0,), dtype=_INDICES_DTYPE)
    lookup_indices = np.empty((0,), dtype=_INDICES_DTYPE)
    lookup_image_path = np.empty((0,), dtype=np.object)

    desc_factory = cv2.ORB_create(config.nfeatures)

    count = 0
    for index, zip_info in batch_iterable:
        count += 1

        if zip_info.is_dir():
            continue

        # unzip in memory
        bytes = config.zip_file.read(zip_info)

        # create image from data
        image = np.frombuffer(bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _, descs = desc_factory.detectAndCompute(image, None)
        if descs is None:
            continue

        features = np.append(features,
                            descs,
                            axis=0)
        indices = np.append(indices,
                            [index] * descs.shape[0],
                            axis=0)
        lookup_indices = np.append(lookup_indices,
                                   [index],
                                   axis=0)
        lookup_image_path = np.append(lookup_image_path,
                                      [zip_info.filename.encode(_LOOKUP_ENCODING)],
                                      axis=0)

    if features.shape[0] > 0:
        part_name = str(uuid.uuid4())
        h5_file.create_dataset(part_name,
                               data=features)

        part_index = part_name + "-index"
        h5_file.create_dataset(part_index,
                               data=indices)

        part_lookup_index = part_name + "-lookup_index"
        h5_file.create_dataset(part_lookup_index,
                               data=lookup_indices)

        part_lookup_image_path = part_name + "-lookup_image_path"
        h5_file.create_dataset(part_lookup_image_path,
                               data=lookup_image_path)

        parts = (part_name, part_index, features.shape[0],
                 part_lookup_index, part_lookup_image_path, lookup_indices.shape[0])
    else:
        parts = None

    return count, parts

def _extract_features_done(results, accum):
    count, parts = results
    accum.progress.update(count),

    if not parts is None:
        part_name, part_index, part_features_count, \
        part_lookup_index, part_lookup_image_path, part_lookup_count = parts

        accum.features_count += part_features_count
        accum.features_parts.append(part_name)
        accum.indices_parts.append(part_index)

        accum.lookup_count += part_lookup_count
        accum.lookup_indices.append(part_lookup_index)
        accum.lookup_image_paths.append(part_lookup_image_path)

def _extract_threaded(config):
    with h5py.File(config.install_path, "w") as h5_file:
        with tqdm(total=config.image_path_count) as progress:
            results = MetaObject.from_kwargs(
                features_parts=[],
                indices_parts=[],
                features_count=0,
                lookup_indices=[],
                lookup_image_paths=[],
                lookup_count=0,
                progress=progress)

            def generate_iterable():
                for index, image_path in enumerate(config.image_path_iterable):
                    yield index, config.zip_file.getinfo(image_path)

            parallel_for(generate_iterable(),
                        _extract_features,
                        config,
                        h5_file,
                        task_completed=lambda r: _extract_features_done(r, results),
                        executor=config.executor,
                        chunk_size=config.chunk_size)

        #
        # merge results into 1 dataset
        #
        features_layout = h5py.VirtualLayout((results.features_count, _FEATURES_WIDTH),
                                             dtype=_FEATURES_DTYPE)

        indices_layout = h5py.VirtualLayout((results.features_count,),
                                            dtype=_INDICES_DTYPE)

        start = 0
        for part_name, part_index in zip(results.features_parts,
                                         results.indices_parts):
            feature_part = h5_file[part_name]
            index_part = h5_file[part_index]

            stop = start + feature_part.shape[0]

            features_layout[start:stop, ...] = h5py.VirtualSource(feature_part)
            indices_layout[start:stop] = h5py.VirtualSource(index_part)

            start = stop

        h5_file.create_virtual_dataset(_FEATURES_KEY, features_layout)
        h5_file.create_virtual_dataset(_INDICES_KEY, indices_layout)


        lookup_indices_layout = h5py.VirtualLayout((results.lookup_count,),
                                                   dtype=_INDICES_DTYPE)
        lookup_image_paths_layout = h5py.VirtualLayout((results.lookup_count,),
                                                       dtype=h5py.string_dtype(encoding=_LOOKUP_ENCODING))
        start = 0
        for lookup_indices, lookup_image_paths in zip(results.lookup_indices,
                                                      results.lookup_image_paths):
            lookup_indices_part = h5_file[lookup_indices]
            lookup_image_paths_part = h5_file[lookup_image_paths]

            stop = start + lookup_indices_part.shape[0]

            lookup_indices_layout[start:stop] = h5py.VirtualSource(lookup_indices_part)
            lookup_image_paths_layout[start:stop] = h5py.VirtualSource(lookup_image_paths_part)

            start = stop

        h5_file.create_virtual_dataset(_LOOKUP_INDICES_KEY, lookup_indices_layout)
        h5_file.create_virtual_dataset(_LOOKUP_IMAGE_PATH_KEY, lookup_image_paths_layout)


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
        self.image_path_iterable = None
        self.image_path_count = 0
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

        # in case we want to inspect from which image each feature comes from
        h5_indices = h5_file[_INDICES_KEY]
        h5_lookup_indices = h5_file[_LOOKUP_INDICES_KEY]
        h5_lookup_image_paths = h5_file[_LOOKUP_IMAGE_PATH_KEY]
        lookup_object = MetaObject.from_kwargs(indices=h5_lookup_indices,
                                               image_paths=h5_lookup_image_paths)

        return MetaObject.from_kwargs(h5_file=h5_file,
                                      features=h5_features,
                                      indices=h5_indices,
                                      lookup=lookup_object)

    if not config.force_generate and \
       os.path.exists(config.install_path):
        return instantiate()

    if config.force_generate or \
       not os.path.exists(config.install_path):
        # s'assurer que le folder destination existe
        dataset_path, _ = os.path.split(config.install_path)
        os.makedirs(dataset_path, exist_ok=True)

    display_html("<b>Extraire ORB features</b>")
    _extract_threaded(config)

    return instantiate()
