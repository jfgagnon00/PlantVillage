import h5py
import os
import pickle

from .BoVWConfig import BoVWConfig
from .DatasetVWConfig import DatasetVWConfig
from .ClassifierConfig import ClassifierConfig
from .processing import _preprocess_bag_model, \
                        _batch_extract_parallel, \
                        _get_idf, \
                        _VISUAL_WORDS_FREQS_KEY, \
                        _TRAIN_VISUAL_WORDS_FREQS_KEY, \
                        _TEST_VISUAL_WORDS_FREQS_KEY, \
                        _INDEX_TO_VISUAL_WORDS_FREQS_KEY
from ..MetaObject import MetaObject


def _write_bag(config, idf, model):
    with open(config.install_path, "wb") as pickled_file:
        pickle.dump((idf, model), pickled_file)

def _instantiate_bag(config):
    with open(config.install_path, "rb") as pickled_file:
        (idf, model) = pickle.load(pickled_file)

    return MetaObject.from_kwargs(model=model,
                                  idf=idf,
                                  cluster_centers=model["kmeans"].cluster_centers_)

def _instantiate_dataset(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)
    h5_vw_freqs = h5_file[_VISUAL_WORDS_FREQS_KEY]
    h5_train_vw_freqs = h5_file[_TRAIN_VISUAL_WORDS_FREQS_KEY]
    h5_test_vw_freqs = h5_file[_TEST_VISUAL_WORDS_FREQS_KEY]
    h5_index_to_vw_freqs = h5_file[_INDEX_TO_VISUAL_WORDS_FREQS_KEY]

    return MetaObject.from_kwargs(h5_file=h5_file,
                                  vw_freqs=h5_vw_freqs,
                                  train_vw_freqs=h5_train_vw_freqs,
                                  test_vw_freqs=h5_test_vw_freqs,
                                  index_to_vw_freqs=h5_index_to_vw_freqs)

def load_bovw(config, features):
    """
    Utilitaire encapsulant extraction/loading du dictionnaire.

    config:
        Instance de BoVWConfig

    features:
        features servant a construire le dictionnaire

    Retour:
        MetaObject encapsulant le dictionnaire. Si le fichier demande existe,
        il est retourne sinon il est construit a partir de features.
    """
    if not config.force_generate and os.path.exists(config.install_path):
        return _instantiate_bag(config)

    if config.force_generate or not os.path.exists(config.install_path):
        path, file  = os.path.split(config.install_path)
        if not file is None:
            os.makedirs(path, exist_ok=True)

    print("Construction Bag of Visual Words")
    model = _preprocess_bag_model(config, features)
    _write_bag(config, None, model)

    return _instantiate_bag(config)

def load_dataset_vw(dataset_vw_config,
                    features,
                    bovw_config,
                    bovw_metaobject,
                    train_indices,
                    test_indices):
    """
    Utilitaire encapsulant extraction/loading en batch des
    visual words d'un dataset.

    config:
        Instance de DatasetVWConfig

    features:
        MetaObject features servant a construire les visual words

    bovw_metaobject:
        MetaObject retourne par load_bovw()

    train_indices, test_indices:
        Iterateur sur index desires

    Retour:
        MetaObject encapsulant le tous les visual words d'un dataset. Si le
        fichier demande existe, il est retourne sinon il est construit.
    """
    if not dataset_vw_config.force_generate and \
        os.path.exists(dataset_vw_config.install_path):
        return _instantiate_dataset(dataset_vw_config)

    if dataset_vw_config.force_generate or \
        not os.path.exists(dataset_vw_config.install_path):
        path, file  = os.path.split(dataset_vw_config.install_path)
        if not file is None:
            os.makedirs(path, exist_ok=True)

    print("Construction Visual Words")
    with h5py.File(dataset_vw_config.install_path, "w") as h5_file:
        _batch_extract_parallel(dataset_vw_config,
                                features,
                                bovw_metaobject.model,
                                bovw_metaobject.cluster_centers.shape[0],
                                train_indices,
                                test_indices,
                                h5_file)
        bovw_metaobject.idf = _get_idf(h5_file)
        _write_bag(bovw_config, bovw_metaobject.idf, bovw_metaobject.model)

    return _instantiate_dataset(dataset_vw_config)
