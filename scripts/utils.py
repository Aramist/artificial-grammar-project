import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
from dynamax.hidden_markov_model import CategoricalHMM
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


def sample_rollouts(
    n: int,
    sequence_length: int,
    initial_probs: jax.Array,
    transition_matrix: jax.Array,
    emission_matrix: jax.Array,
    rng: jax.random.PRNGKey,
) -> tp.Tuple[jax.Array, jax.Array]:
    """Samples sequences of observations and hidden states from a hidden Markov model.

    Args:
        n (int): Number of sequences to sample
        sequence_length (int): Length of each sequence
        initial_probs (jax.Array): Initial probability distribution over hidden states. Shape (num_states,)
        transition_matrix (jax.Array): Hidden state transition probability matrix. Shape (num_states, num_states)
        emission_matrix (jax.Array): Probability distributions of observed variable given hidden states. Shape (num_states, emission_dim, num_classes)
        rng (jax.random.PRNGKey): Random state

    Returns:
        jax.Array: Hidden states of shape (n, sequence_length) in range [0, num_states)
        jax.Array: Observations of shape (n, sequence_length, emission_dim) in range [0, num_classes)
        jax.Array: Optimal predictions of state obtained via filtration (n, sequence_length, emission_dim, num_classes)
    """

    num_states = initial_probs.shape[0]
    if emission_matrix.ndim == 2:
        num_emissions = 1
        emission_matrix = emission_matrix[:, None, :]
    else:
        num_emissions = emission_matrix.shape[1]
    num_classes = emission_matrix.shape[-1]

    hmm = CategoricalHMM(num_states, num_emissions, num_classes)

    hmm_params, _ = hmm.initialize(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_probs=emission_matrix,
    )

    keys = jax.random.split(rng, n)
    sampler = jax.vmap(hmm.sample, in_axes=(None, 0, None), out_axes=0)  # map over keys

    states, emissions = sampler(hmm_params, keys, sequence_length)

    posterior = jax.vmap(hmm.filter, in_axes=(None, 0), out_axes=0)(
        hmm_params, emissions
    )  # map over emissions
    # Posteriors are of shape (n, sequence_length, num_states)
    # Multiply with emissions matrix to get probabilities of each class
    posterior_class = jnp.einsum(
        "bts,sec->btec", posterior.filtered_probs, emission_matrix
    )

    return states, emissions, posterior_class


if __name__ == "__main__":
    # Test
    checkpoint_dir = Path("checkpoints/transformer_model_5000_samples").absolute()
    model: transformerlib.TransformerLM = load_model(
        checkpoint_dir, transformerlib.TransformerLM
    )
    params = nnx.state(model, nnx.Param)
    total_params = sum(
        jnp.prod(jnp.array(x.shape)) for x in jax.tree_util.tree_leaves(params)
    )
    print(f"Loaded transformer with {total_params} parameters.")
    print(
        f"Num heads: {model.num_heads}, Num layers: {model.num_layers}, d_model: {model.d_model}"
    )

    states, emissions, optimal_posterior = sample_rollouts(
        n=10,
        sequence_length=512,
        initial_probs=jnp.ones((2,)) / 2,
        transition_matrix=jnp.array(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ]
        ),
        emission_matrix=jnp.array([[1 / 6] * 6, [1 / 10] * 5 + [1 / 2]]),
        rng=jax.random.key(0),
    )

    emissions = emissions.squeeze(-1)  # transformer expects (n, seq_len) input

    transformer_posterior = model(emissions)
    # Check that the intermediates were sown properly
    assert hasattr(model.transformer.layers[0], "attn_weights")
