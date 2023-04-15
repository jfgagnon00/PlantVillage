import h5py
import os

from .Config import Config
from ..MetaObject import MetaObject


def _instantiate(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)

    return MetaObject.from_kwargs(h5_file=h5_file)

def load(config, features_iter):
    """
    Utilitaire encapsulant extraction/loading du dictionnaire.

    config:
        Instance de Config

    features_iter:
        iterateur sur numpy array

    Retour:
        MetaObject encapsulant le dictionnaire. Si le fichier demande existe,
        il est retourne sinon il est construit a partir de features_iter.
    """
    if not config.force_generate and os.path.exists(config.install_path):
        return _instantiate(config)

    if config.force_generate or not os.path.exists(config.install_path):
        path, file  = os.path.split(config.install_path)
        if not file is None:
            os.makedirs(path, exist_ok=True)

    print("Constrution Bag of Visual Words")
    with h5py.File(config.install_path, "w") as h5_file:
        pass

    return _instantiate(config)
