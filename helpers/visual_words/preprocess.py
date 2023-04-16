from os  import cpu_count
from sklearn.cluster import MiniBatchKMeans


def _preprocess(config, features):
    mb_kmeans = MiniBatchKMeans(n_clusters=config.n_clusters, batch_size=256 * cpu_count())
    return mb_kmeans.fit(features)
