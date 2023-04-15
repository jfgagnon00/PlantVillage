class Config():
    """
    Parametres configurant l'installation/preprocessing
    du dictionnaire
    """
    def __init__(self, executor=None):
        self.install_path = "dataset/BoVW.hd5"
        self.read_only = True
