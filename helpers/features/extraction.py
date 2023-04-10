import cv2
import h5py
import numpy as np
import uuid

from tqdm.notebook import tqdm

from .key_points import _cv_key_points_to_list

from ..Concurrent import parallel_for
from ..MetaObject import MetaObject


_FEATURES_KEY = "features"
_INDICES_KEY = "indices"
_INDICES_DTYPE = np.uint16

_KEYPOINTS_KEY = "keypoints"
_KEYPOINTS_WIDTH = 7
_KEYPOINTS_DTYPE = np.float32


def _batch_extract(config,
                   h5_file,
                   batch_iterables):
    features = np.empty((0, config.features_width()), dtype=config.features_dtype())
    key_points = np.empty((0, _KEYPOINTS_WIDTH), dtype=_KEYPOINTS_DTYPE)
    indices = np.empty((0,), dtype=_INDICES_DTYPE)

    desc_factory = config.create_factory()

    count = 0
    for index, _, image_future in batch_iterables:
        count += 1

        image = image_future()
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

def _batch_done(batch_results, progress, batch_accum):
    count, batch = batch_results

    progress.update(count)

    if not batch is None:
        batch_accum.features_count += batch.features_count
        batch_accum.features_ds_names.append(batch.features_ds_name)
        batch_accum.key_points_ds_names.append(batch.key_points_ds_name)
        batch_accum.indices_ds_names.append(batch.indices_ds_name)

def _batch_merge(config, h5_file, batch_accum):
    features_layout = h5py.VirtualLayout((batch_accum.features_count, config.features_width()),
                                         dtype=config.features_dtype())

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

def _batch_extract_parallel(config, h5_file, dataset_iter):
    batch_accum = MetaObject.from_kwargs(
        features_count=0,
        features_ds_names=[],
        key_points_ds_names=[],
        indices_ds_names=[])

    with tqdm(total=dataset_iter.count) as progress:
        parallel_for(dataset_iter,
                     _batch_extract,
                     config,
                     h5_file,
                     task_completed=lambda r: _batch_done(r, progress, batch_accum),
                     executor=config.executor,
                     chunk_size=config.chunk_size)

    _batch_merge(config, h5_file, batch_accum)
