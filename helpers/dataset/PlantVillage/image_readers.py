import cv2
import numpy as np

def _get_image_zip_info(zip_file, zip_info):
    image = zip_file.read(zip_info)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def _get_image_filename(zip_file, filename):
    return _get_image_zip_info(zip_file, zip_file.getinfo(filename))

def _get_thumbnail_filename(h5_file, filename):
    return h5_file[filename][...]
