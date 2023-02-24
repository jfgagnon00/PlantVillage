import os
import requests
import time
import zipfile

from jupyter_helpers import display_html
from tqdm.notebook import tqdm


def _get_default(kwargs, key, default_value):
    return default_value if kwargs is None else kwargs.get(key, default_value)

def download_with_progress(dest_path, 
                           url, 
                           skip_download=False):
    """
    Utilitaire pour afficher progres d'un download
    
    dest_path:
        Chemin pour enregistrer resultat du download.
        
    url:
        Url du fichier a downloader. Assumer fichier zip
        
    skip_download:
        Pour fin de debug. Skip download.
    
    Retour:
        Nom du fichier downloader ou None si download echoue.
    """
    try:
        r = requests.get(url, stream=True)

        content_size = int(r.headers.get('content-length'))
        content_type = r.headers.get('content-type').lower()

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
    
def unzip_with_progress(dest_path, 
                        filename, 
                        unzip_one_folder_up=True,
                        skip_extract=False):
    """
    Utilitaire pour afficher progres sur unzip d'un fichier
    
    dest_path:
        Chemin pour dossier ou sera installer dataset
        
    file:
        Chemin du fichier a dezipper
        
    unzip_one_folder_up:
        False: extrait zip a l'emplacement dicte par file
        True: extrait zip un path plus haut
        
    skip_extract:
        Pour fin de debug. Skip zip extraction.
        
    Retour:
        True si succes, False autrement
    """
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
        
def install_dataset(dest_path, dataset_url, **kwargs):
    """
    Utilitaire pour installer dataset.    
    
    dest_path: 
        Chemin pour dossier ou sera installer dataset
    
    dataset_url: 
        URL dataset. Attendu que ce soit 1 fichier .zip
        
    kwargs:
        Info supplementaire pour diriger installation. Voir 
        implementation pour details.
    
    Retour: 
        True si installation reussie, False autrement.
    """
    display_html(f"<b>Downloading</b> <i>{dataset_url}</i>")
    skip_download = _get_default(kwargs, "skip_download", False)
    zip_file = download_with_progress(dest_path, 
                                      dataset_url, 
                                      skip_download=skip_download)
    if zip_file is None:
        display_html(f"<b>Failed</b>")
        return False
    
    display_html(f"<b>Unzipping</b> <i>{zip_file}</i>")
    unzip_one_folder_up = _get_default(kwargs, "unzip_one_folder_up", True)
    skip_extract = _get_default(kwargs, "skip_extract", False)
    if not unzip_with_progress(dest_path,  
                               zip_file, 
                               unzip_one_folder_up=unzip_one_folder_up,
                               skip_extract=skip_extract):
        display_html("<b>Failed</b>")
        return False
    
    display_html("<b>Cleaning</b>")
    os.remove(zip_file)
    
    return True