class DatasetIter():
    """
    Bridge les dataset et l'extraction de features.
    Itere sur une paire (index, image_path) et retourne une promise
    pour lire l'image representee par la paire
    """
    def __init__(self,
                 dataset,
                 iterable_index_imagepath,
                 iterable_count=-1):
        self._get_image = dataset.get_image
        self._iterable_index_imagepath = iterable_index_imagepath
        self._iterable_count = iterable_count

    @property
    def count(self):
        return self._iterable_count

    def __iter__(self):
        return self

    def __next__(self):
        index, image_path = next(self._iterable_index_imagepath)
        return index, image_path, lambda: self._get_image(index)
