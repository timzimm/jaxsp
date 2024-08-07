import math
import hashlib
import os
import pickle


import jax.numpy as jnp
import mgzip

# import orbax.checkpoint as ocp

# checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler(), timeout_secs=50)


def name_of_stacked_names(names):
    # Re-hash because model might be stacked (due to vmap)
    combined = hashlib.sha256()
    combined.update(hashlib.md5(names).digest())
    return hash_to_int64(combined.hexdigest())


def save_model(enforce_save, cache_dir, name, model):
    path = f"{cache_dir}/{name}"
    if os.path.exists(path) and not enforce_save:
        return
    if os.path.exists(path):
        os.remove(path)
    # save_args = ocp.args.StandardSave(model)
    # checkpointer.save(path, model, save_args=save_args)
    with mgzip.open(path, mode="wb") as file:
        pickle.dump(model, file)


def load_model(cache_dir, name):
    path = f"{cache_dir}/{name}"
    if os.path.exists(path):
        # checkpointer.restore(path)
        # jax.tree_util.tree_map
        # logger.info(f"Model already computed. Load {name}...")
        with mgzip.open(f"{cache_dir}/{name}", mode="rb") as file:
            model = pickle.load(file)
            return model
    return None


def load_or_compute_model(load_if_available, cache_dir, name, compute, *args):
    model = None
    if load_if_available:
        model = load_model(cache_dir, name)
    if not model:
        # logger.info(f"Model {name} not found/Recompute enforced...")
        return compute(*args)
    return model


def hash_to_int64(digested_hash):
    num = int(digested_hash, 16)
    num = num // 10 ** (int(math.log(num, 10)) - 18 + 1)
    return jnp.int64(num)
