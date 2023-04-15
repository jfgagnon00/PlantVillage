class Config():
    """
    Parametres configurant l'installation/preprocessing
    du code book
    """
    def __init__(self, executor=None):
        self.install_path = "dataset/CodeBook.hd5"
        self.read_only = True
