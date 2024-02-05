import hashlib
import os
import pickle
import logging
import ruamel.yaml

import jax.numpy as jnp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def name_of_stacked_names(names):
    # Re-hash because model might be stacked (due to vmap)
    combined = hashlib.sha256()
    combined.update(hashlib.md5(names).digest())
    return hash_to_int32(combined.hexdigest())


def save_model(cache_dir, name, model):
    logger.info(f"Save {name}...")
    with open(f"{cache_dir}/{name}", mode="wb") as file:
        pickle.dump(model, file)


def load_model(cache_dir, name):
    if os.path.exists(f"{cache_dir}/{name}"):
        logger.info(f"Model already computed. Load {name}...")
        with open(f"{cache_dir}/{name}", mode="rb") as file:
            model = pickle.load(file)
            return model
    return None


def load_or_compute_model(load_if_available, cache_dir, name, compute, *args):
    model = load_model(cache_dir, name)
    if not model:
        logger.info(f"Model {name} not found/Recompute enforced...")
        return compute(*args)
    return model


def hash_to_int32(digested_hash):
    name_full_str = "".join(str(ord(c)) for c in digested_hash)
    return jnp.int32(name_full_str[:10])


def load_parameters_from_config(path_to_file):
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(path_to_file, "r") as file:
        parameters = yaml.load(file)

    check_parameters(parameters)
    return parameters


def check_parameters(parameters):
    pass
