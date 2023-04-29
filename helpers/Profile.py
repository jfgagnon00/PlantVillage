from time import time


class Profile():
    """
    Utilitaire pour mesurer la duree d'un bout de code.
    Utilisation attendues est:

    with Profile() as pro:
        do_your_stuff()

    print( pro.round_duration() )
    """
    def __init__(self):
        self._enter = time()
        self._exit = time()

    def __enter__(self):
        self._exit = time()
        return self

    def __exit__(self, type, value, traceback):
        self._exit = time()

    @property
    def duration(self):
        return self._exit - self._enter

    def round_duration(self, digits=2):
        return round(self.duration, digits)