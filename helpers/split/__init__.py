import csv


def load(config):
    """
    Utilitaire pour lire les ensembles train/test

    config:
        Instance de Config

    retour:
        Tuple (train, test). Chaque tuple est une liste
        d'index
    """
    with open(config.train_install_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        train = next(csv_reader)
        train = [int(v) for v in train]

    with open(config.test_install_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        test = next(csv_reader)
        test = [int(v) for v in test]

    return train, test

def save(config, train, test):
    """
    Utilitaire pour ecrire les ensembles train/test

    config:
        Instance de Config

    train, test:
        Liste des d'index pour les ensemble de train/test
    """
    with open(config.train_install_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(train)

    with open(config.test_install_path, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(test)
