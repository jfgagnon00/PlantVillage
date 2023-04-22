import cv2
import h5py
import numpy as np
import uuid

from tqdm.notebook import tqdm

from .key_points import _cv_key_points_to_list

from ..Concurrent import parallel_for
from ..MetaObject import MetaObject


_FEATURES_KEY = "features"

_KEYPOINTS_KEY = "keypoints"
_KEYPOINTS_WIDTH = 7
_KEYPOINTS_DTYPE = np.float32

_INDICES_PREFIX = "indices"

_TRAIN_FEATURES_KEY = "train_features"
_TEST_FEATURES_KEY = "test_features"


def _extract(desc_factory, image):
    kpts, descs = desc_factory.detectAndCompute(image, None)
    if descs is None:
        return None, None
    else:
        return _cv_key_points_to_list(kpts), descs

def _batch_extract(config,
                   h5_file,
                   batch_iterables):
    features = np.empty((0, config.features_width()), dtype=config.features_dtype())
    key_points = np.empty((0, _KEYPOINTS_WIDTH), dtype=_KEYPOINTS_DTYPE)
    indices = []

    desc_factory = config.create_factory()

    batch_start = 0
    count = 0
    for index, _, image_future in batch_iterables:
        count += 1

        image = image_future()
        kpts, descs = _extract(desc_factory, image)
        if descs is None:
            continue

        batch_stop = batch_start + descs.shape[0]
        features = np.append(features,
                            descs,
                            axis=0)

        key_points = np.append(key_points,
                               kpts,
                               axis=0)

        indices.append( (index, batch_start, batch_stop) )
        batch_start = batch_stop

    if features.shape[0] > 0:
        batch_name = str(uuid.uuid4())
        features_ds = h5_file.create_dataset(batch_name,
                                             data=features)

        batch_key_points = batch_name + "-key_points"
        key_points_ds = h5_file.create_dataset(batch_key_points,
                                               data=key_points)

        # map each index for later fast query
        for index, batch_start, batch_stop in indices:
            features_layout = h5py.VirtualLayout((batch_stop - batch_start, config.features_width()), dtype=config.features_dtype())
            key_points_layout = h5py.VirtualLayout((batch_stop - batch_start, _KEYPOINTS_WIDTH), dtype=_KEYPOINTS_DTYPE)

            features_layout[...] = h5py.VirtualSource(features_ds)[batch_start:batch_stop, ...]
            key_points_layout[...] = h5py.VirtualSource(key_points_ds)[batch_start:batch_stop, ...]

            h5_file.create_virtual_dataset(f"{_INDICES_PREFIX}/{_FEATURES_KEY}/{index}", features_layout)
            h5_file.create_virtual_dataset(f"{_INDICES_PREFIX}/{_KEYPOINTS_KEY}/{index}", key_points_layout)

        batch = MetaObject.from_kwargs(
                features_count=features.shape[0],
                features_ds_name=batch_name,
                key_points_ds_name=batch_key_points)
    else:
        batch = None

    return count, batch

def _batch_done(batch_results, progress, batch_accum):
    count, batch = batch_results

    progress.update(count)

    if not batch is None:
        batch_accum.features_count += batch.features_count
        batch_accum.features_ds_names.append(batch.features_ds_name)
        batch_accum.key_points_ds_names.append(batch.key_points_ds_name)

def _batch_merge(config, h5_file, batch_accum):
    features_layout = h5py.VirtualLayout((batch_accum.features_count, config.features_width()),
                                         dtype=config.features_dtype())

    key_points_layout = h5py.VirtualLayout((batch_accum.features_count, _KEYPOINTS_WIDTH),
                                           dtype=_KEYPOINTS_DTYPE)

    start = 0
    for feature_name, \
        key_point_name in zip(batch_accum.features_ds_names,
                              batch_accum.key_points_ds_names):
        features = h5_file[feature_name]
        key_points = h5_file[key_point_name]

        stop = start + features.shape[0]

        features_layout[start:stop, ...] = h5py.VirtualSource(features)
        key_points_layout[start:stop, ...] = h5py.VirtualSource(key_points)

        start = stop

    h5_file.create_virtual_dataset(_FEATURES_KEY, features_layout)
    h5_file.create_virtual_dataset(_KEYPOINTS_KEY, key_points_layout)

def _batch_extract_parallel(config, h5_file, dataset_iter):
    batch_accum = MetaObject.from_kwargs(
        features_count=0,
        features_ds_names=[],
        key_points_ds_names=[])

    with tqdm(total=dataset_iter.count) as progress:
        parallel_for(dataset_iter,
                     _batch_extract,
                     config,
                     h5_file,
                     task_completed=lambda r: _batch_done(r, progress, batch_accum),
                     executor=config.executor,
                     chunk_size=config.chunk_size)

    _batch_merge(config, h5_file, batch_accum)

def _update_virtual_features_dataset(h5_file, name, indices):
    index_to_features = f"{_INDICES_PREFIX}/{_FEATURES_KEY}"

    # amasser toutes les sources indiquer par indices
    virtual_sources = []
    features_count = 0
    for i in indices:
        i_str = str(i)
        if i_str in h5_file[index_to_features]:
            ds = h5_file[index_to_features][i_str]
            features_count += ds.shape[0]

            # https://docs.h5py.org/en/stable/high/dataset.html#reference
            vs = ds.virtual_sources()

            virtual_sources.extend(vs)

    # en faire un dataset virtuel
    features_ds = h5_file[_FEATURES_KEY]
    dataset_layout = h5py.VirtualLayout((features_count, features_ds.shape[1]), dtype=features_ds.dtype)

    offset = 0
    for vs in virtual_sources:
        # https://api.h5py.org/h5s.html#h5py.h5s.SpaceID
        source_ds = h5_file[vs[2]]
        source_start, source_end = vs[3].get_select_bounds()
        source_start, source_end = source_start[0], source_end[0]
        source_size = source_end - source_start + 1

        dataset_layout[offset:offset + source_size, ...] = h5py.VirtualSource(source_ds)[source_start:source_start + source_size, ...]

        offset += source_size

    if name in h5_file:
        del h5_file[name]

    h5_file.create_virtual_dataset(name, dataset_layout)
