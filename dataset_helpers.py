import cv2
import json
import math
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


def _get_default(kwargs, key, default_value):
    return default_value if kwargs is None else kwargs.get(key, default_value)

def _download_with_progress(dest_path,
                           url,
                           skip_download=False):
    try:
        r = requests.get(url, stream=True)

        content_size = int(r.headers.get('content-length'))
        content_disposition = r.headers.get('content-disposition')
        filename = content_disposition.split("=", 1)[-1]
        filename = filename.replace('"', "")
        filename = os.path.join(dest_path, filename)

        if not skip_download:
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

def _unzip_with_progress(dest_path,
                        filename,
                        unzip_one_folder_up=True,
                        skip_extract=False):
    try:
        with zipfile.ZipFile(file=filename) as zip_file:
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
                    # pour eviter message d'erreur dans jupyter notebook
                    time.sleep(0.01)
                    continue

                if unzip_one_folder_up:
                    zip_info.filename = one_up_path

                if not skip_extract:
                    zip_file.extract(zip_info, path=dest_path)

            progress.refresh()
    except Exception as e:
        print(e)
        return False
    else:
        return True

def _dataset_install(dataset_config, **kwargs):
    display_html(f"<b>Downloading</b> <i>{dataset_config.url}</i>")
    skip_download = _get_default(kwargs, "skip_download", False)
    zip_file = _download_with_progress(dataset_config.install_path,
                                       dataset_config.url,
                                       skip_download=skip_download)
    if zip_file is None:
        display_html(f"<b>Failed</b>")
        return False

    display_html(f"<b>Unzipping</b> <i>{zip_file}</i>")
    unzip_one_folder_up = _get_default(kwargs, "unzip_one_folder_up", True)
    skip_extract = _get_default(kwargs, "skip_extract", False)
    if not _unzip_with_progress(dataset_config.install_path,
                                zip_file,
                                unzip_one_folder_up=unzip_one_folder_up,
                                skip_extract=skip_extract):
        display_html("<b>Failed</b>")
        return False

    display_html("<b>Cleaning</b>")
    os.remove(zip_file)

    return True

def _gather_files(dataset_config):
    dataset_files = os.path.join(dataset_config.install_path, "*", "*.*")
    thumbbail_path = os.path.join(dataset_config.install_path, ".thumbnails")
    dataset_files = [os.path.normpath(f)
                     for f in iglob(dataset_files)
                        if not thumbbail_path in f]
    return list(dataset_files), thumbbail_path

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
                          dataset_files,
                          thumbbail_path):
    progress = tqdm(iterable=dataset_files,
                    total=len(dataset_files),
                    bar_format="{l_bar}{bar}{postfix}")

    for image_path in progress:
        _, labels, file = image_path.split(os.sep)

        progress.set_postfix_str( os.path.join(labels, file) )

        image = cv2.imread(image_path)
        h, w, _ = image.shape
        th = math.ceil(h * dataset_config.thumbnail_scale)
        tw = math.ceil(w * dataset_config.thumbnail_scale)
        thumbnail = cv2.resize(image, (tw, th))

        thumbnail_folder = os.path.join(thumbbail_path, labels)
        thumbnail_file = os.path.join(thumbnail_folder, file)
        os.makedirs(thumbnail_folder, exist_ok=True)
        cv2.imwrite(thumbnail_file, thumbnail)

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
        Voir _dataset_install

    Retour:
        Pandas.DataFrame representant le
        dataset ou None si probleme.
    """
    def dataset_read():
        df = read_csv(dataset_config.preprocess_path,
                      quoting=QUOTE_NONNUMERIC)

        # see https://tinyurl.com/3uywpnxp for details
        df.image_width = df.image_width.astype(int)
        df.image_height = df.image_height.astype(int)

        return df

    if os.path.exists(dataset_config.preprocess_path):
        return dataset_read()

    if dataset_config.install and \
       not os.path.exists(dataset_config.install_path):
        os.makedirs(dataset_config.install_path)

        display_html(f"<b>Installing Dataset</b>")
        if not _dataset_install(dataset_config, kwargs=kwargs):
            display_html(f"<b>Dataset installation error</b>")
        else:
            display_html(f"<b>Dataset installed</b>")
    else:
        display_html(f"<b>Dataset already installed</b>")

    dataset_files, thumbbail_path = _gather_files(dataset_config)

    if not os.path.exists(thumbbail_path):
        display_html("<b>Dataset preprocessing thumbnails</b>")
        _preprocess_thumbnail(dataset_config,
                                      dataset_files,
                                      thumbbail_path)
    else:
        display_html("<b>Dataset thumbnails already preprocessed</b>")

    display_html("<b>Dataset preprocessing csv</b>")
    _preprocess_csv(dataset_config,
                            dataset_files,
                            thumbbail_path)

    return dataset_read()
