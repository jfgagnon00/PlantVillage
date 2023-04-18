from os  import cpu_count
from sklearn.cluster import MiniBatchKMeans


def _preprocess_bag_model(config, features):
    mb_kmeans = MiniBatchKMeans(n_clusters=config.n_clusters, batch_size=256 * cpu_count())
    return mb_kmeans.fit(features)


def _batch_extract_parallel(config,
                            features,
                            bovw_model,
                            dataset_iter,
                            h5_file):
    pass
