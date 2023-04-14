import h5py
import os
import requests

from pandas import DataFrame
from zipfile import ZipFile
from tqdm.notebook import tqdm

from .Config import Config
from .image_readers import _get_image_filename, \
                           _get_thumbnail_filename
from .preprocess import _preprocess_parallel, \
                        _DATAFRAME_KEY, \
                        _DATAFRAME_SOURCE_ATTR, \
                        _DATAFRAME_COLUMNS

from ...MetaObject import MetaObject


def _instantiate(config):
    mode = "r" if config.read_only else "r+"

    h5_file = h5py.File(config.install_path, mode)
    h5_dataframe = h5_file[_DATAFRAME_KEY]
    dataframe = DataFrame({c:h5_dataframe[c].asstr()[...] for c in _DATAFRAME_COLUMNS})

    path, file = os.path.split(config.install_path)
    if file is None:
        zip_filename = h5_dataframe.attrs[_DATAFRAME_SOURCE_ATTR]
    else:
        zip_filename = os.path.join(path, h5_dataframe.attrs[_DATAFRAME_SOURCE_ATTR])
    zip_file = ZipFile(zip_filename)

    return MetaObject.from_kwargs(h5_file=h5_file,
                                  dataframe=dataframe,
                                  get_image=lambda index: _get_image_filename(zip_file, \
                                                                              dataframe.loc[index, "image_path"]),
                                  get_thumbnail=lambda index: _get_thumbnail_filename(h5_file, \
                                                                                      dataframe.loc[index, "thumbnail_path"]))

def _download(config, dest_path):
    try:
        r = requests.get(config.url, stream=True)

        content_size = int(r.headers.get('content-length'))
        content_disposition = r.headers.get('content-disposition')
        filename = content_disposition.split("=", 1)[-1]
        filename = filename.replace('"', "")
        filename = os.path.join(dest_path, filename)

        if config.force_download or not os.path.exists(filename):
            with open(filename, "wb") as f:
                with tqdm(total=content_size) as progress:
                    for data in r.iter_content(chunk_size=16*1024):
                        f.write(data)
                        progress.update(len(data))
    except Exception as e:
        print(e)
        return None
    else:
        return filename


def load(config):
    """
    Utilitaire encapsulant installation et preprocessing du dataset PlantVillage.

    config:
        Instance de Config

    Retour:
        MetaObject encapsulant le dataset ou None si probleme.
    """
    if not config.force_install and os.path.exists(config.install_path):
        return _instantiate(config)

    if config.force_install or not os.path.exists(config.install_path):
        print(f"Downloading {config.url}")

        path, file = os.path.split(config.install_path)
        if file is None:
            path = ""
        else:
            os.makedirs(path, exist_ok=True)

        zip_filename = _download(config, path)
        if zip_filename is None:
            print("Failed")
            return None

        print(f"Preprocesssing {zip_filename}")
        _preprocess_parallel(config, zip_filename)

    return _instantiate(config)
