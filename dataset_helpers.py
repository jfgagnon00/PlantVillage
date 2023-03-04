import cv2
import h5py
import json
import math
import numpy as np
import os
import re
import requests
import time
import zipfile

from csv import QUOTE_NONNUMERIC
from glob import iglob
from jupyter_helpers import display_html
from pandas import DataFrame, read_csv
from tqdm.notebook import tqdm


_IMAGES_KEY = "images"
_THUMBNAILS_KEY = "thumbnail"
_FEATURES_KEY = "features"


def _get_default(kwargs, key, default_value):
    return default_value if kwargs is None else kwargs.get(key, default_value)

def _download_with_progress(dest_path,
                           url,
                           force_download):
    try:
        r = requests.get(url, stream=True)

        content_size = int(r.headers.get('content-length'))
        content_disposition = r.headers.get('content-disposition')
        filename = content_disposition.split("=", 1)[-1]
        filename = filename.replace('"', "")
        filename = os.path.join(dest_path, filename)

        if force_download or not os.path.exists(filename):
            with open(filename, "wb") as f:
                progress = tqdm(total=content_size)
                for data in r.iter_content(chunk_size=16*1024):
                    f.write(data)
                    progress.update(len(data))
                progress.refresh()
    except Exception as e:
        print(e)
        return None
    else:
        return filename

def _h5extract_with_progress(h5_filename,
                             zip_filename,
                             force_extract,
                             extract_one_folder_up):
    try:
        with h5py.File(h5_filename, "a") as h5_file:
            if _IMAGES_KEY in h5_file:
                if force_extract:
                    del h5_file[_IMAGES_KEY]
                else:
                    return True

            h5_images = h5_file.create_group(_IMAGES_KEY)
            with zipfile.ZipFile(file=zip_filename) as zip_file:
                infolist = zip_file.infolist()
                progress = tqdm(iterable=infolist,
                                total=len(infolist),
                                bar_format="{l_bar}{bar}{postfix}")

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
                        time.sleep(0.01)
                        continue

                    if extract_one_folder_up:
                        zip_info.filename = one_up_path

                    # zip_info.filename = os.path.join(h5_images.name, zip_info.filename)
                    zip_info.filename = zip_info.filename.replace("\\", "/")
                    zip_bytes = zip_file.read(zip_info)

                    image_np = np.frombuffer(zip_bytes, dtype=np.uint8)
                    image_np = cv2.imdecode(image_np, cv2.COLOR_BGR2RGB)

                    h5_images.create_dataset(zip_info.filename,
                                             data=image_np)

                progress.refresh()
    except Exception as e:
        print(e)
        return False
    else:
        return True

def _dataset_install(dataset_config,
                     force_download=False,
                     keep_download=False,
                     force_extract=False,
                     extract_one_folder_up=True,
                     **kwargs):
    display_html(f"<b>Downloading</b> <i>{dataset_config.url}</i>")

    dataset_path, _ = os.path.split(dataset_config.install_path)
    os.makedirs(dataset_path, exist_ok=True)
    zip_file = _download_with_progress(dataset_path,
                                       dataset_config.url,
                                       force_download)
    if zip_file is None:
        display_html(f"<b>Failed</b>")
        return False

    display_html(f"<b>Extracting</b> <i>{zip_file}</i>")
    if not _h5extract_with_progress(dataset_config.install_path,
                                    zip_file,
                                    force_extract,
                                    extract_one_folder_up):
        display_html("<b>Failed</b>")
        return False

    display_html("<b>Cleaning</b>")

    if not keep_download:
        os.remove(zip_file)

    return True

def _preprocess_csv(dataset_config,
                    dataset_files,
                    thumbbail_path):
    data = []
    species_disease_pattern = re.compile(dataset_config.species_disease_re)
    progress = tqdm(iterable=dataset_files,
                    total=len(dataset_files),
                    bar_format="{l_bar}{bar}{postfix}")

    for image_path in progress:
        _, labels, file = image_path.split(os.sep)
        sub_path = os.path.join(labels, file)

        progress.set_postfix_str(sub_path)

        labels_match = species_disease_pattern.match(labels)
        if labels_match is None:
            continue

        plant_species, plant_disease = labels_match.groups()

        image = cv2.imread(image_path)
        h, w, _ = image.shape
        thumbnail_file = os.path.join(thumbbail_path, sub_path)

        # some elements do not respect nomenclature
        # found in litterature: fix it
        species_match = species_disease_pattern.match(plant_species)
        if not species_match is None:
            plant_species = dataset_config.label_separator.join(species_match.groups())

        if plant_species in plant_disease:
            label = plant_disease
        else:
            label = dataset_config.label_separator.join([plant_species,
                                                         plant_disease])

        data.append( (label,
                      image_path,
                      plant_species,
                      plant_disease,
                      w,
                      h,
                      thumbnail_file) )

    progress.update()

    preprocessed_df = DataFrame(data,
                                columns=["label",
                                         "image_path",
                                         "species",
                                         "disease",
                                         "image_width",
                                         "image_height",
                                         "thumbnail_path"])
    preprocessed_df.to_csv(dataset_config.preprocess_path,
                           index=False,
                           quoting=QUOTE_NONNUMERIC)

def _preprocess_thumbnail(dataset_config,
                          h5_file,
                          force_thumbnail_generation=False,
                          **kwargs):
    if _THUMBNAILS_KEY in h5_file:
        if force_thumbnail_generation:
            del h5_file[_THUMBNAILS_KEY]
        else:
            return

    h5_thumbnails = h5_file.create_group(_THUMBNAILS_KEY)
    images = h5_file[_IMAGES_KEY].items()
    progress = tqdm(iterable=images,
                    total=len(images),
                    bar_format="{l_bar}{bar}{postfix}")




    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            # node is a dataset
        else:
            # node is a group


    for labels, image in progress:
        print(labels)
        print(image.name.split("/"))
        print()
        # print(list(image.keys()))

        progress.set_postfix_str(image.name)

        # image_np = image[...]
        # h, w, _ = image_np.shape
        # th = math.ceil(h * dataset_config.thumbnail_scale)
        # tw = math.ceil(w * dataset_config.thumbnail_scale)
        # thumbnail = cv2.resize(image_np, (tw, th))
        # h5_thumbnails.create_dataset(labels,
        #                              data=thumbnail)

    h5_file.flush()
    progress.update()

def dataset_load_config(filename):
    """
    Converti un fichier json en un objet

    filename:
        Nom du fichier json a lire.

    Retour:
        Objet representant le fichier json ou None si probleme.
    """
    class config_obj(object):
        def __init__(self, dict_):
            self.__dict__.update(dict_)

    try:
        with open(filename) as f:
            config = json.load(f, object_hook=config_obj)
    except Exception as e:
        print(e)
        return None
    else:
        return config

def dataset_load(dataset_config, **kwargs):
    """
    Utilitaire encapsulant installation et
    preprocessing d'un dataset

    dataset_config:
        Object retourne par dataset_load_config()

    kwargs:
        Voir _dataset_install et _preprocess_thumbnail

    Retour:
        object encapsulant le dataset ou
        None si probleme.
    """
    def dataset_read():
        df = read_csv(dataset_config.preprocess_path,
                      quoting=QUOTE_NONNUMERIC)

        # see https://tinyurl.com/3uywpnxp for details
        df.image_width = df.image_width.astype(int)
        df.image_height = df.image_height.astype(int)

        return df

    if not dataset_config.force_install and \
       os.path.exists(dataset_config.install_path):
        return dataset_read()

    if dataset_config.force_install or \
       not os.path.exists(dataset_config.install_path):
        display_html(f"<b>Installing Dataset</b>")
        if _dataset_install(dataset_config, **kwargs):
            display_html(f"<b>Dataset installed</b>")
        else:
            display_html(f"<b>Dataset installation error</b>")
    else:
        display_html(f"<b>Dataset already installed</b>")

    with h5py.File(dataset_config.install_path, "r+") as h5_file:
        display_html("<b>Dataset preprocessing thumbnails</b>")
        _preprocess_thumbnail(dataset_config,
                              h5_file,
                              **kwargs)

        # display_html("<b>Dataset preprocessing csv</b>")
        # _preprocess_csv(dataset_config,
        #                 dataset_files,
        #                 thumbbail_path)

        h5_file.flush()

    return dataset_read()
