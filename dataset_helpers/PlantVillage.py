import cv2
import h5py
import math
import os
import re
import requests
import zipfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from jupyter_helpers import display_html
from .MetaObject import MetaObject
from numpy import frombuffer
from tqdm.notebook import tqdm


_IMAGES_KEY = "images"
_IMAGE_COUNT_ATTR = "image_count"

_THUMBNAILS_KEY = "thumbnails"
_THUMBNAIL_COUNT_ATTR = "thumbnail_count"

_DATAFRAME_KEY = "dataframe"
_DATAFRAME_COLUMNS_COUNT_ATTR = "dataframe_columns_count"
_DATAFRAME_COLUMNS = ["species",
                      "disease",
                      "image_path",
                      "thumbnail_path",
                      "label"]
_DATAFRAME_ENCODING = "utf-8"


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

def _initialize_h5(h5_file):
    h5_images = h5_file.create_group(_IMAGES_KEY)
    h5_thumbnails = h5_file.create_group(_THUMBNAILS_KEY)

    h5_dataframe = h5_file.create_group(_DATAFRAME_KEY)
    h5_dataframe.attrs[_DATAFRAME_COLUMNS_COUNT_ATTR] = len(_DATAFRAME_COLUMNS)

    dataframe_data = {k: [] for k in _DATAFRAME_COLUMNS}

    return MetaObject.from_dict(
           {
                "images": h5_images,
                "thumbnails": h5_thumbnails,
                "dataframe": h5_dataframe,
                "dataframe_data": dataframe_data
           })

def _update_h5_dataframe(h5_meta):
    for column_name in _DATAFRAME_COLUMNS:
        data = h5_meta.dataframe_data[column_name]
        h5_meta.dataframe.create_dataset(column_name,
                                         data=data,
                                         dtype=h5py.string_dtype(encoding=_DATAFRAME_ENCODING))

def _update_h5(h5_meta,
               name,
               image,
               thumbnail,
               dataframe):
    h5_meta.images.create_dataset(name, data=image)
    h5_meta.thumbnails.create_dataset(name, data=thumbnail)

    if not dataframe is None:
        for key, value in dataframe.items():
            h5_meta.dataframe_data[key].append(value)

def _extract_dataframe(config, filename):
    species_disease_pattern = re.compile(config.species_disease_re)
    species_pattern = re.compile(config.species_re)

    labels = filename.split("/")[-2]
    labels_match = species_disease_pattern.match(labels)
    if labels_match is None:
        return None

    plant_species, plant_disease = labels_match.groups()

    # some elements do not respect nomenclature
    # found in litterature: fix it
    species_match = species_pattern.match(plant_species)
    if not species_match is None:
        plant_species = config.label_separator.join(species_match.groups())

    # some elements duplicate plant species within plan disease
    # keep plant species only 1x
    if plant_species in plant_disease:
        label = plant_disease
    else:
        label = config.label_separator.join([plant_species,
                                             plant_disease])

    return {
        "species": plant_species.encode(_DATAFRAME_ENCODING),
        "disease": plant_disease.encode(_DATAFRAME_ENCODING),
        "image_path": "/".join([_IMAGES_KEY, filename]).encode(_DATAFRAME_ENCODING),
        "thumbnail_path": "/".join([_THUMBNAILS_KEY, filename]).encode(_DATAFRAME_ENCODING),
        "label": label.encode(_DATAFRAME_ENCODING)
    }

def _extract(config, zip_file, zip_info):
    if zip_info.is_dir():
        return None

    filename = zip_info.filename.replace("\\", "/")
    if config.extract_one_folder_up:
        filename = filename.split("/")[1:]
        filename = "/".join(filename)

    # dataframe
    dataframe = _extract_dataframe(config, filename)
    if dataframe is None:
        return None

    # unzip in memory
    bytes = zip_file.read(zip_info)

    # create image from data
    image = frombuffer(bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # scale down
    h, w, _ = image.shape
    th = math.ceil(h * config.thumbnail_scale)
    tw = math.ceil(w * config.thumbnail_scale)
    thumbnail = cv2.resize(image, (tw, th))

    return filename, \
           image, \
           thumbnail, \
           dataframe

def _extract_threaded(config, zip_filename):
    with h5py.File(config.install_path, "w") as h5_file:
        h5_meta = _initialize_h5(h5_file)

        with zipfile.ZipFile(zip_filename) as zip_file:
            zip_infos = zip_file.infolist()
            worker_count = os.cpu_count() // 2
            image_count = 0

            with ThreadPoolExecutor(max_workers=worker_count) as thread_pool:
                futures = [thread_pool.submit(_extract,
                                            config,
                                            zip_file,
                                            zip_info) for zip_info in zip_infos]

                with tqdm(total=len(zip_infos)) as progress:
                    for f in as_completed(futures):
                        progress.update()

                        results = f.result()
                        if not results is None:
                            image_count += 1
                            _update_h5(h5_meta, *results)

        _update_h5_dataframe(h5_meta)

        h5_meta.images.attrs[_IMAGE_COUNT_ATTR] = image_count
        h5_meta.thumbnails.attrs[_THUMBNAIL_COUNT_ATTR] = image_count
        h5_file.flush()

class PlantVillageConfig():
    """
    Parametres configurant l'installation/preprocessing
    de PlantVillage
    """
    def __init__(self):
        self.url = "https://tinyurl.com/22tas3na"
        self.install_path = "dataset/PlantVillage.hd5"
        self.species_disease_re = "(.*)(?:___)(.*)"
        self.species_re = "(.*)(?:,_|_)(.*)"
        self.label_separator = "_"
        self.thumbnail_scale = 0.25

        self.force_download = False
        self.keep_download = False
        self.extract_one_folder_up = True

def plant_village_load(config):
    """
    Utilitaire encapsulant installation et
    preprocessing d'un dataset

    config:
        Instance of PlantVillageConfig

    Retour:
        MetaObject encapsulant le dataset ou
        None si probleme.
    """
    def instantiate():
        h5_file = h5py.File(config.install_path, "r")
        h5_dataframe = h5_file[_DATAFRAME_KEY]

        attrs = {}
        attrs["h5_file"] = h5_file
        attrs["data"] = {c:h5_dataframe[c].asstr()[...] for c in _DATAFRAME_COLUMNS}

        return MetaObject.from_dict(attrs)

    if not config.force_install and \
       os.path.exists(config.install_path):
        return instantiate()

    if config.force_install or \
        not os.path.exists(config.install_path):
        display_html(f"<b>Downloading</b> <i>{config.url}</i>")

        # s'assurer que le folder destination existe
        dataset_path, _ = os.path.split(config.install_path)
        os.makedirs(dataset_path, exist_ok=True)

        zip_file = _download(config, dataset_path)
        if zip_file is None:
            display_html(f"<b>Failed</b>")
            return None

    display_html(f"<b>Extracting</b> <i>{zip_file}</i>")
    _extract_threaded(config, zip_file)

    if not config.keep_download:
        display_html("<b>Cleaning</b>")
        os.remove(zip_file)

    return instantiate()
