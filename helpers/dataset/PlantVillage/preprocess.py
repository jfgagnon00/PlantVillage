import cv2
import h5py
import math
import os
import re

from tqdm.notebook import tqdm
from zipfile import ZipFile

from .image_readers import _get_image_zip_info

from ...Concurrent import parallel_for
from ...MetaObject import MetaObject


_THUMBNAILS_KEY = "thumbnails"

_DATAFRAME_KEY = "dataframe"
_DATAFRAME_SOURCE_ATTR = "dataframe_source"
_DATAFRAME_COLUMNS = ["species",
                      "disease",
                      "label",
                      "image_path",
                      "thumbnail_path"]
_DATAFRAME_ENCODING = "utf-8"


def _initialize_h5(h5_file, zip_basename):
    h5_thumbnails = h5_file.create_group(_THUMBNAILS_KEY)

    h5_dataframe = h5_file.create_group(_DATAFRAME_KEY)
    h5_dataframe.attrs[_DATAFRAME_SOURCE_ATTR] = zip_basename.encode(_DATAFRAME_ENCODING)

    dataframe_data = {k: [] for k in _DATAFRAME_COLUMNS}

    return MetaObject.from_dict({
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
               thumbnail,
               dataframe):
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
        "label": label.encode(_DATAFRAME_ENCODING),
        "image_path": filename.encode(_DATAFRAME_ENCODING),
        "thumbnail_path": "/".join([_THUMBNAILS_KEY, filename]).encode(_DATAFRAME_ENCODING)
    }

def _preprocess(config, zip_file, zip_info):
    if zip_info.is_dir():
        return None

    filename = zip_info.filename.replace("\\", "/")

    dataframe = _extract_dataframe(config, filename)
    if dataframe is None:
        return None

    image = _get_image_zip_info(zip_file, zip_info)

    h, w, _ = image.shape
    th = math.ceil(h * config.thumbnail_scale)
    tw = math.ceil(w * config.thumbnail_scale)
    thumbnail = cv2.resize(image, (tw, th))

    return filename, thumbnail, dataframe

def _preprocess_parallel(config, zip_filename):
    with h5py.File(config.install_path, "w") as h5_file:
        path, file = os.path.split(zip_filename)
        zip_basename = path if file is None else file
        h5_meta = _initialize_h5(h5_file, zip_basename)

        with ZipFile(zip_filename) as zip_file:
            zip_infos = zip_file.infolist()

            def _preprocess_completed(results):
                progress.update()

                if not results is None:
                    _update_h5(h5_meta, *results)

            with tqdm(total=len(zip_infos)) as progress:
                parallel_for(zip_infos,
                    _preprocess,
                    config,
                    zip_file,
                    task_completed=_preprocess_completed,
                    executor=config.executor)

        _update_h5_dataframe(h5_meta)
