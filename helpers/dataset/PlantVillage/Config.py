class Config():
    """
    Parametres configurant l'installation/preprocessing
    de PlantVillage
    """
    def __init__(self, executor=None):
        self.url = "https://tinyurl.com/22tas3na"
        self.install_path = "dataset/PlantVillage.hd5"
        self.species_disease_re = "(.*)(?:___)(.*)"
        self.species_re = "(.*)(?:,_|_)(.*)"
        self.label_separator = "_"
        self.thumbnail_scale = 0.25

        self.force_download = False
        self.read_only = True
        self.executor = executor
