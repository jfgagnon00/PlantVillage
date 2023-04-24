class Config():
    """
    Parametres configurant l'installation/preprocessing
    de PlantVillage
    """
    def __init__(self, executor=None):
        self.url = ""
        self.install_path = ""
        self.species_disease_re = "(.*)(?:___)(.*)"
        self.species_re = "(.*)(?:,_|_)(.*)"
        self.label_separator = "_"

        self.train_install_path = ""
        self.test_install_path = ""

        self.force_download = False
        self.force_install = False
        self.read_only = True
        self.executor = executor

        
        self.thumbnail_scale = 0.25