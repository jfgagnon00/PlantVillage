class BoVWConfig():
    """
    Parametres configurant l'installation/preprocessing
    du dictionnaire (Bag of Visual Words)
    """
    def __init__(self):
        self.install_path = ""
        self.force_generate = True
        self.kmeans_n_clusters = 200
        self.pca_n_components = None
