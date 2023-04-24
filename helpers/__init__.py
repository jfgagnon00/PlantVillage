from .Concurrent import create_thread_pool_executor
from .MetaObject import MetaObject
from .Profile import Profile
from .Jupyter import display_html


def get_configs(config_overrides, executor=None):
    """
    Utilitaire pour obtenir la configuration complete du pipeline.

    config_overrides:
        Nom du fichier .json qui contient les overrides des configs.

    executor:
        Pour les etapes qui requierent du multiprocessing, l'executor
        a utiliser. Si None, Concurrent.create_thread_pool_executor()
        sera utilise.

    retour:
        MetaObject contenant toute la configuration avec les overrides
        appliques.
    """
    if executor is None:
        executor = create_thread_pool_executor(max_workers=None)

    try:
        config_overrides = MetaObject.from_json(config_overrides)
    except:
        config_overrides = None
        pass

    from .dataset.PlantVillage.Config import Config as PlantVillageConfig
    from .features import OrbFeaturesConfig, SiftFeaturesConfig
    from .split.Config import Config as SplitConfig
    from .visual_words import BoVWConfig, DatasetVWConfig

    # generer les configs par defaut
    pv_config = PlantVillageConfig(executor)

    orb_config = OrbFeaturesConfig(executor)
    orb_bovw_config = BoVWConfig()
    orb_ds_vw_config = DatasetVWConfig(executor)

    sift_config = SiftFeaturesConfig(executor)
    sift_bovw_config = BoVWConfig()
    sift_ds_vw_config = DatasetVWConfig(executor)

    split_config = SplitConfig()

    # appliquer les overrides
    if not orb_config is None:
        MetaObject.override_from_object(pv_config,
                                        config_overrides.dataset)

        MetaObject.override_from_object(orb_config,
                                        config_overrides.orb.features)

        MetaObject.override_from_object(orb_bovw_config,
                                        config_overrides.orb.bovw)

        MetaObject.override_from_object(orb_ds_vw_config,
                                        config_overrides.orb.dataset_vw)

        MetaObject.override_from_object(sift_config,
                                        config_overrides.sift.features)

        MetaObject.override_from_object(sift_bovw_config,
                                        config_overrides.sift.bovw)

        MetaObject.override_from_object(sift_ds_vw_config,
                                        config_overrides.sift.dataset_vw)

        MetaObject.override_from_object(split_config,
                                        config_overrides.split)

    return MetaObject.from_kwargs(plant_village=pv_config,
                                  orb=orb_config,
                                  orb_bovw=orb_bovw_config,
                                  orb_dataset_vw=orb_ds_vw_config,
                                  sift=sift_config,
                                  sift_bovw=sift_bovw_config,
                                  sift_dataset_vw=sift_ds_vw_config,
                                  split=split_config)
