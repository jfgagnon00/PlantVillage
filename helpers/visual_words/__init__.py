import os
import pickle

from .BoVWConfig import BoVWConfig
from .preprocess import _preprocess
from .VisualWords import VisualWords
from ..MetaObject import MetaObject


def _instantiate_bag(config):
    with open(config.install_path, "rb") as pickled_file:
        model = pickle.load(pickled_file)

    # TODO: ajouter des attribut wrapper?
    return MetaObject.from_kwargs(model=model)

def load_bovw(config, features):
    """
    Utilitaire encapsulant extraction/loading du dictionnaire.

    config:
        Instance de BoVWConfig

    features:
        data servant a construire le dictionnaire

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
    model = _preprocess(config, features)
    with open(config.install_path, "wb") as pickled_file:
        pickle.dump(model, pickled_file)

    return _instantiate_bag(config)
