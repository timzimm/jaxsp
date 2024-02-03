import os
import pickle
import logging

import jax.numpy as jnp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_and_save_model(cache_dir, name, compute, *compute_args):
    model = compute(*compute_args)
    with open(f"{cache_dir}/{name}", mode="wb") as file:
        pickle.dump(model, file)
    return model


def load_or_compute_model(load_if_available, cache_dir, name_of, compute, *args):
    name = name_of(*args)
    if os.path.exists(f"{cache_dir}/{name}") and load_if_available:
        logger.info(f"Model already computed. Load {name}...")
        with open(f"{cache_dir}/{name}", mode="rb") as file:
            return pickle.load(file)
    else:
        logger.info("No model found/Recompute enforced. Compute it...")
        return compute_and_save_model(cache_dir, name, compute, *args)


def hash_to_int32(digested_hash):
    name_full_str = "".join(str(ord(c)) for c in digested_hash)
    return jnp.int32(name_full_str[:10])
