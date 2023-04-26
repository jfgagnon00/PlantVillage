import h5py
import numpy as np
import uuid

from os  import cpu_count
from sklearn.cluster import MiniBatchKMeans
from tqdm.notebook import tqdm

from ..Concurrent import parallel_for
from ..MetaObject import MetaObject

_VISUAL_WORDS_FREQS_KEY = "vw_freqs/all"
_TRAIN_VISUAL_WORDS_FREQS_KEY = "vw_freqs/train"
_TEST_VISUAL_WORDS_FREQS_KEY = "vw_freqs/test"
_INDEX_TO_VISUAL_WORDS_FREQS_KEY = "indices/vw_freqs"
_IDF_KEY = "vw_freqs/train_idf"

_VISUAL_WORDS_FREQS_TYPE = np.float16


def _preprocess_bag_model(config, features):
    mb_kmeans = MiniBatchKMeans(n_clusters=config.n_clusters,
                                batch_size=256 * cpu_count(),
                                n_init="auto")
    return mb_kmeans.fit(features)

def _extract(bovw_model, n_clusters, features_array):
    vw_freq = bovw_model.predict(features_array)
    vw_freq = np.bincount(vw_freq, minlength=n_clusters)
    vw_freq = vw_freq / float(n_clusters)
    vw_freq = vw_freq.astype(_VISUAL_WORDS_FREQS_TYPE)

    return vw_freq

def _batch_extract(features,
                   bovw_model,
                   h5_file,
                   indices_iterables):
    n_clusters = bovw_model.cluster_centers_.shape[0]

    vw_freqs = np.empty((0, n_clusters), dtype=_VISUAL_WORDS_FREQS_TYPE)
    indices = []

    count = 0
    for index in indices_iterables:
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
            h5_file.create_virtual_dataset(f"{_INDEX_TO_VISUAL_WORDS_FREQS_KEY}/{index}", layout)

        batch = MetaObject.from_kwargs(vw_count=vw_freqs.shape[0],
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
    layout = h5py.VirtualLayout((batch_accum.vw_count, n_clusters), dtype=_VISUAL_WORDS_FREQS_TYPE)

    start = 0
    for name in batch_accum.vw_ds_name:
        batch = h5_file[name]
        stop = start + batch.shape[0]
        layout[start:stop, ...] = h5py.VirtualSource(batch)
        start = stop

    h5_file.create_virtual_dataset(_VISUAL_WORDS_FREQS_KEY, layout)

def _ordered_dataset(ds_name, h5_file, indices, n_clusters):
    index_to_vs = h5_file[_INDEX_TO_VISUAL_WORDS_FREQS_KEY]

    layout = h5py.VirtualLayout((len(indices), n_clusters), dtype=_VISUAL_WORDS_FREQS_TYPE)

    start = 0
    for index in indices:
        batch = index_to_vs[ str(index) ]
        stop = start + batch.shape[0]
        layout[start:stop, ...] = h5py.VirtualSource(batch)
        start = stop

    h5_file.create_virtual_dataset(ds_name, layout)

def _batch_extract_parallel(config,
                            features,
                            bovw_model,
                            train_indices,
                            test_indices,
                            h5_file):
    batch_accum = MetaObject.from_kwargs(
        vw_count=0,
        vw_ds_name=[])

    all_indices = train_indices + test_indices

    with tqdm(total=len(all_indices)) as progress:
        parallel_for(all_indices,
                     _batch_extract,
                     features,
                     bovw_model,
                     h5_file,
                     task_completed=lambda r: _batch_done(r, progress, batch_accum),
                     executor=config.executor,
                     chunk_size=config.chunk_size)

    n_clusters = bovw_model.n_clusters
    _batch_merge(h5_file, batch_accum, n_clusters)

    _ordered_dataset(_TRAIN_VISUAL_WORDS_FREQS_KEY, h5_file, train_indices, n_clusters)
    _ordered_dataset(_TEST_VISUAL_WORDS_FREQS_KEY, h5_file, test_indices, n_clusters)

def _get_idf(dataset_vw_h5_file):
    vw_freqs = dataset_vw_h5_file[_TRAIN_VISUAL_WORDS_FREQS_KEY]

    idf = np.count_nonzero(vw_freqs, axis=0)
    idf = float(1 + vw_freqs.shape[0]) / (idf + 1)
    idf = np.log(idf) + 1.0
    idf = idf.astype(np.float32)
    idf = np.expand_dims(idf, axis=0)

    return idf