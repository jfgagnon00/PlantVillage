import numpy as np

from os  import cpu_count
from sklearn.cluster import MiniBatchKMeans
from tqdm.notebook import tqdm

from ..Concurrent import parallel_for


_VW_DTYPE = np.int32


def _preprocess_bag_model(config, features):
    mb_kmeans = MiniBatchKMeans(n_clusters=config.n_clusters, batch_size=256 * cpu_count())
    return mb_kmeans.fit(features)

# def _extract(desc_factory, image):
#     kpts, descs = desc_factory.detectAndCompute(image, None)
#     if descs is None:
#         return None, None
#     else:
#         return _cv_key_points_to_list(kpts), descs

def _batch_extract(features,
                   bovw_model,
                   h5_file,
                   batch_iterables):
    # features = np.empty((0, config.features_width()), dtype=config.features_dtype())

    count = 0
    for index, _, _ in batch_iterables:
        count += 1

        iii = np.where(features.indices == index)[0]
        print( type(iii) )
        print( iii.shape )

        # indices = np.append(indices,
        #                     [index] * descs.shape[0],
        #                     axis=0)

    if features.shape[0] > 0:
        pass

    return count

def _batch_extract_parallel(config,
                            features,
                            bovw_model,
                            dataset_iter,
                            h5_file):
    with tqdm(total=dataset_iter.count) as progress:
        parallel_for(dataset_iter,
                     _batch_extract,
                     features,
                     bovw_model,
                     h5_file,
                     task_completed=lambda count: progress.update(count),
                     executor=config.executor,
                     chunk_size=config.chunk_size)
