import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp

import rnnlib
import transformerlib


def load_model(checkpoint_dir: Path, model_class: tp.Type[nnx.Module]) -> nnx.Module:
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    # Get the metadata first to determine the structure of the model
    metadata = checkpointer.restore(
        checkpoint_dir, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())
    ).metadata
    # Rngs are not stored with the rest of the state
    rngs = nnx.Rngs(
        params=jax.random.key(0),
        dropout=jax.random.key(1),
    )
    model = model_class(**metadata, rngs=rngs)
    # Now load the parameters and update the model
    _, _, cur_state = nnx.split(model, nnx.RngState, ...)
    restored_state = checkpointer.restore(
        checkpoint_dir,
        args=ocp.args.Composite(state=ocp.args.PyTreeRestore(cur_state)),
    ).state
    nnx.update(model, restored_state)

    return model


if __name__ == "__main__":
    # Test
    checkpoint_dir = Path("checkpoints/transformer_model_5000_samples").absolute()
    model = load_model(checkpoint_dir, transformerlib.TransformerLM)
