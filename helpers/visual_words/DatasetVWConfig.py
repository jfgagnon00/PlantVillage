class DatasetVWConfig():
    """
    Parametres configurant le preprocessing
    des visual words pour un dataset complet
    """
    def __init__(self, executor, chunk_size=150):
        self.install_path = ""
        self.force_generate = True
        self.read_only = True
        self.executor = executor
        self.chunk_size = chunk_size
