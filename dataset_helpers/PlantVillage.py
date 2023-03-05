import cv2
import h5py
import json
import math
import numpy as np
import os
import re
import requests
import zipfile

from jupyter_helpers import display_html
from .MetaObject import MetaObject
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
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

def _extract_h5(config, zip_filename):
    try:
        with h5py.File(config.install_path, "a") as h5_file:
            if _IMAGES_KEY in h5_file:
                if config.force_extract:
                    del h5_file[_IMAGES_KEY]
                else:
                    return True

            h5_images = h5_file.create_group(_IMAGES_KEY)
            with zipfile.ZipFile(file=zip_filename) as zip_file:
                h5_image_count = 0
                infolist = zip_file.infolist()

                with tqdm(iterable=infolist,
                          total=len(infolist),
                          bar_format="{l_bar}{bar}{postfix}") as progress:
                    for zip_info in progress:
                        # display filename in progress bar
                        # but without start folder
                        path, file = os.path.split(zip_info.filename)
                        _, path = os.path.split(path)
                        one_up_path = os.path.join(path, file)

                        progress.set_postfix_str(one_up_path)

                        if zip_info.is_dir():
                            # pour eviter message d'erreur dans
                            # jupyter notebook
                            sleep(0.01)
                            continue

                        if config.extract_one_folder_up:
                            zip_info.filename = one_up_path

                        zip_info.filename = zip_info.filename.replace("\\", "/")
                        zip_bytes = zip_file.read(zip_info)

                        image_np = np.frombuffer(zip_bytes, dtype=np.uint8)
                        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                        h5_image_count += 1
                        h5_images.create_dataset(zip_info.filename,
                                                data=image_np)

            h5_images.attrs[_IMAGE_COUNT_ATTR] = h5_image_count

    except Exception as e:
        print(e)
        return False
    else:
        return True

def _install(config):
    display_html(f"<b>Downloading</b> <i>{config.url}</i>")

    dataset_path, _ = os.path.split(config.install_path)
    os.makedirs(dataset_path, exist_ok=True)
    zip_file = _download(config, dataset_path)
    if zip_file is None:
        display_html(f"<b>Failed</b>")
        return False

    display_html(f"<b>Extracting</b> <i>{zip_file}</i>")
    if not _extract_h5(config, zip_file):
        display_html("<b>Failed</b>")
        return False

    if not config.keep_download:
        display_html("<b>Cleaning</b>")
        os.remove(zip_file)

    return True

def _preprocess_dataframe(config, h5_file,
                          force_dataframe_generation=False,
                          **kwargs):
    if _DATAFRAME_KEY in h5_file:
        if force_dataframe_generation:
            del h5_file[_DATAFRAME_KEY]
        else:
            return

    species_disease_pattern = re.compile(config.species_disease_re)

    h5_images = h5_file[_IMAGES_KEY]
    h5_image_count = h5_images.attrs[_IMAGE_COUNT_ATTR]

    h5_dataframe = h5_file.create_group(_DATAFRAME_KEY)
    h5_dataframe.attrs[_DATAFRAME_COLUMNS_COUNT_ATTR] = len(_DATAFRAME_COLUMNS)

    def create_dataset(column_name):
        return h5_dataframe.create_dataset(column_name,
                                           shape=(0,),
                                           dtype=h5py.string_dtype(encoding="utf-8"),
                                           maxshape=(None,))

    def append(key, value):
        dataset = h5_dataframe[key]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1:] = value.encode("utf-8")

    [create_dataset(c) for c in _DATAFRAME_COLUMNS]

    with tqdm(total=h5_image_count,
              bar_format="{l_bar}{bar}{postfix}") as progress:
        def features(h5_node):
            labels, file = h5_node.name.split("/")[2:]

            progress.set_postfix_str("/".join([labels, file]))
            progress.update()

            labels_match = species_disease_pattern.match(labels)
            if labels_match is None:
                return

            plant_species, plant_disease = labels_match.groups()

            # some elements do not respect nomenclature
            # found in litterature: fix it
            species_match = species_disease_pattern.match(plant_species)
            if not species_match is None:
                plant_species = config.label_separator.join(species_match.groups())

            # some elements duplicate plant species within plan disease
            # keep plant species only 1x
            if plant_species in plant_disease:
                label = plant_disease
            else:
                label = config.label_separator.join([plant_species,
                                                             plant_disease])

            thumbnail_path = "/".join(["", _THUMBNAILS_KEY, labels, file])

            append("species", plant_species)
            append("disease", plant_disease)
            append("image_path", h5_node.name)
            append("thumbnail_path", thumbnail_path)
            append("label", label)

        # mucho slow
        h5_images.visititems(lambda _, o: features(o) if isinstance(o, h5py.Dataset) else None)






def _initialize_h5(h5_file):
    h5_images = h5_file.create_group(_IMAGES_KEY)
    h5_thumbnails = h5_file.create_group(_THUMBNAILS_KEY)

    h5_dataframe = h5_file.create_group(_DATAFRAME_KEY)
    h5_dataframe.attrs[_DATAFRAME_COLUMNS_COUNT_ATTR] = len(_DATAFRAME_COLUMNS)

    for column_name in _DATAFRAME_COLUMNS:
        h5_dataframe.create_dataset(column_name,
                                    shape=(0,),
                                    dtype=h5py.string_dtype(encoding="utf-8"),
                                    maxshape=(None,))

    return MetaObject.from_dict(
           {
                "images": h5_images,
                "thumbnails": h5_thumbnails,
                "dataframe": h5_dataframe,
           })

def _update_h5(h5_meta,
               name,
               image,
               thumbnail,
               dataframe):
    h5_meta.images.create_dataset(name, data=image)
    h5_meta.thumbnails.create_dataset(name, data=thumbnail)

    if not dataframe is None:
        for key, value in dataframe.items():
            dataset = h5_meta.dataframe[key]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1:] = value.encode("utf-8")

def _extract_dataframe(config,
                       species_disease_pattern,
                       filename):
    labels = filename.split("/")[-2]
    labels_match = species_disease_pattern.match(labels)
    if labels_match is None:
        return None

    plant_species, plant_disease = labels_match.groups()

    # some elements do not respect nomenclature
    # found in litterature: fix it
    species_match = species_disease_pattern.match(plant_species)
    if not species_match is None:
        plant_species = config.label_separator.join(species_match.groups())

    # some elements duplicate plant species within plan disease
    # keep plant species only 1x
    if plant_species in plant_disease:
        label = plant_disease
    else:
        label = config.label_separator.join([plant_species,
                                             plant_disease])

    return None

    return {
        "species": plant_species,
        "disease": plant_disease,
        "image_path": "/".join([_IMAGES_KEY, filename]),
        "thumbnail_path": "/".join([_THUMBNAILS_KEY, filename]),
        "label": label
    }

def _extract(config,
             species_disease_pattern,
             zip_file,
             zip_info):
    if zip_info.is_dir():
        return None

    filename = zip_info.filename.replace("\\", "/")
    if config.extract_one_folder_up:
        filename = filename.split("/")[1:]
        filename = "/".join(filename)

    # dataframe
    dataframe = _extract_dataframe(config, species_disease_pattern, filename)
    if dataframe is None:
        return None

    # unzip in memory
    bytes = zip_file.read(zip_info)

    # create image from data
    image = np.frombuffer(bytes, dtype=np.uint8)
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
    species_disease_pattern = re.compile(config.species_disease_re)

    with h5py.File(config.install_path, "w") as h5_file:
        h5_meta = _initialize_h5(h5_file)

        with zipfile.ZipFile(zip_filename) as zip_file:
            zip_infos = zip_file.infolist()
            image_count = 0

            with ThreadPoolExecutor() as thread_pool:
                futures = [thread_pool.submit(_extract,
                                            config,
                                            species_disease_pattern,
                                            zip_file,
                                            zip_info) for zip_info in zip_infos]

                with tqdm(total=len(zip_infos)) as progress:
                    for f in as_completed(futures):
                        progress.update()

                        results = f.result()

                        if not results is None:
                            image_count += 1
                            _update_h5(h5_meta, *results)

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
        self.species_disease_re = "(.*)(?:___|_)(.*)"
        self.label_separator = "_"
        self.thumbnail_scale = 0.25

        self.force_download = False
        self.keep_download = False
        self.force_extract = False
        self.extract_one_folder_up = True
        self.force_thumbnail_generation = False
        self.force_dataframe_generation = False

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

        # df = read_csv(config.preprocess_path,
        #               quoting=QUOTE_NONNUMERIC)

        # # see https://tinyurl.com/3uywpnxp for details
        # df.image_width = df.image_width.astype(int)
        # df.image_height = df.image_height.astype(int)

        # return df

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
    if not _extract_threaded(config, zip_file):
        display_html("<b>Failed</b>")
        return None

    if not config.keep_download:
        display_html("<b>Cleaning</b>")
        os.remove(zip_file)



    if False:
        if config.force_install or \
        not os.path.exists(config.install_path):
            display_html(f"<b>Installing Dataset</b>")
            if _install(config):
                display_html(f"<b>Dataset installed</b>")
            else:
                display_html(f"<b>Dataset installation error</b>")
        else:
            display_html(f"<b>Dataset already installed</b>")

        with h5py.File(config.install_path, "r+") as h5_file:
            display_html("<b>Preprocessing thumbnails</b>")
            _preprocess_thumbnail(config, h5_file)

            display_html("<b>Preprocessing dataframe</b>")
            _preprocess_dataframe(config, h5_file)

            h5_file.flush()

    return instantiate()
