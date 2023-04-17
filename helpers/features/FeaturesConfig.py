from abc import ABC, abstractmethod


class FeaturesConfig(ABC):
    """
    Classe abstraite encapsulant les configurations pour
    extraction des features.
    """
    def __init__(self, executor, chunk_size):
        self.install_path = ""
        self.force_generate = False
        self.read_only = True
        self.executor = executor
        self.chunk_size = chunk_size

    @abstractmethod
    def features_width(self):
        pass

    @abstractmethod
    def features_dtype(self):
        pass

    @abstractmethod
    def create_factory(self):
        pass
