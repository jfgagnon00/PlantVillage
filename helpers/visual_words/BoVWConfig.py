class BoVWConfig():
    """
    Parametres configurant l'installation/preprocessing
    du dictionnaire (Bag of Visual Words)
    """
    def __init__(self):
        self.install_path = ""
        self.force_generate = False
        self.n_clusters = 200
