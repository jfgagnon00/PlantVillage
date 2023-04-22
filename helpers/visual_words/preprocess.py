import h5py
import numpy as np
import uuid

from os  import cpu_count
from sklearn.cluster import MiniBatchKMeans
from tqdm.notebook import tqdm

from ..Concurrent import parallel_for
from ..MetaObject import MetaObject


def _preprocess_bag_model(config, features):
    mb_kmeans = MiniBatchKMeans(n_clusters=config.n_clusters, batch_size=256 * cpu_count())
    return mb_kmeans.fit(features)

def _extract(bovw_model, n_clusters, features_array):
    vw_freq = bovw_model.predict(features_array)
    vw_freq = np.bincount(vw_freq, minlength=n_clusters)

    return vw_freq

def _batch_extract(features,
                   bovw_model,
                   h5_file,
                   batch_iterables):
    n_clusters = bovw_model.cluster_centers_.shape[0]

    vw_freqs = np.empty((0, n_clusters), dtype=np.int32)
    indices = []

    count = 0
    for index, _, _ in batch_iterables:
        count += 1

        index_str = str(index)
        if not index_str in features.index_to_features:
            continue

        features_array = features.index_to_features[index_str][...]
        vw_freq = _extract(bovw_model, n_clusters, features_array)
        vw_freq = np.expand_dims(vw_freq, axis=0)

        vw_freqs = np.append(vw_freqs, vw_freq, axis=0)

        indices.append(index)

    if vw_freqs.shape[0] > 0:
        batch_name = str(uuid.uuid4())
        vw_ds = h5_file.create_dataset(batch_name, data=vw_freqs)

        # map each index for later fast query
        for i, index in enumerate(indices):
            layout = h5py.VirtualLayout((1, vw_freqs.shape[1]), dtype=vw_freqs.dtype)
            layout[...] = h5py.VirtualSource(vw_ds)[i, ...]
            h5_file.create_virtual_dataset(f"indices/vw/{index}", layout)

        batch = MetaObject.from_kwargs(
                    vw_count=vw_freqs.shape[0],
                    vw_ds_name=batch_name)
    else:
        batch = None

    return count, batch

def _batch_done(batch_results, progress, batch_accum):
    count, batch = batch_results

    progress.update(count)

    if not batch is None:
        batch_accum.vw_count += batch.vw_count
        batch_accum.vw_ds_name.append(batch.vw_ds_name)

def _batch_merge(h5_file, batch_accum, n_clusters):
    layout = h5py.VirtualLayout((batch_accum.vw_count, n_clusters), dtype=np.int32)

    start = 0
    for name in batch_accum.vw_ds_name:
        batch = h5_file[name]
        stop = start + batch.shape[0]
        layout[start:stop, ...] = h5py.VirtualSource(batch)
        start = stop

    h5_file.create_virtual_dataset("vw", layout)

def _batch_extract_parallel(config,
                            features,
                            bovw_model,
                            dataset_iter,
                            h5_file):
    batch_accum = MetaObject.from_kwargs(
        vw_count=0,
        vw_ds_name=[])

    with tqdm(total=dataset_iter.count) as progress:
        parallel_for(dataset_iter,
                     _batch_extract,
                     features,
                     bovw_model,
                     h5_file,
                     task_completed=lambda r: _batch_done(r, progress, batch_accum),
                     executor=config.executor,
                     chunk_size=config.chunk_size)

    _batch_merge(h5_file, batch_accum, bovw_model.cluster_centers_.shape[0])