import h5py
import os
import pickle

from .BoVWConfig import BoVWConfig
from .DatasetVWConfig import DatasetVWConfig
from .preprocess import _preprocess_bag_model, \
                        _batch_extract_parallel, \
                        _VISUAL_WORDS_FREQS_KEY, \
                        _INDEX_TO_VISUAL_WORDS_FREQS_KEY
from .VisualWords import VisualWords
from ..MetaObject import MetaObject


def _instantiate_bag(config):
    with open(config.install_path, "rb") as pickled_file:
        model = pickle.load(pickled_file)

    # TODO: ajouter des attribut wrapper?
    return MetaObject.from_kwargs(model=model)

def _instantiate_dataset(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)
    h5_vw_freqs = h5_file[_VISUAL_WORDS_FREQS_KEY]
    h5_index_to_vw_freqs = h5_file[_INDEX_TO_VISUAL_WORDS_FREQS_KEY]

    return MetaObject.from_kwargs(h5_file=h5_file,
                                  vw_freqs=h5_vw_freqs,
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
    with open(config.install_path, "wb") as pickled_file:
        pickle.dump(model, pickled_file)

    return _instantiate_bag(config)

def load_dataset_vw(config, features, bovw_model, indices_iter):
    """
    Utilitaire encapsulant extraction/loading en batch des
    visual words d'un dataset.

    config:
        Instance de DatasetVWConfig

    features:
        MetaObject features servant a construire les visual words

    bovw_model:
        model representant le dictionnaire

    indices_iter:
        Iterateur sur index desires

    Retour:
        MetaObject encapsulant le tous les visual words d'un dataset. Si le
        fichier demande existe, il est retourne sinon il est construit.
    """
    if not config.force_generate and os.path.exists(config.install_path):
        return _instantiate_dataset(config)

    if config.force_generate or not os.path.exists(config.install_path):
        path, file  = os.path.split(config.install_path)
        if not file is None:
            os.makedirs(path, exist_ok=True)

    print("Construction Visual Words")
    with h5py.File(config.install_path, "w") as h5_file:
        _batch_extract_parallel(config,
                                features,
                                bovw_model,
                                indices_iter,
                                h5_file)

    return _instantiate_dataset(config)
