import typing as tp
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

import transformerlib
import utils


def grab_diag_window(arr: jax.Array, win_length: int) -> jax.Array:
    def grab_for_row(pad_arr: jax.Array, i: int) -> jax.Array:
        return jax.lax.dynamic_slice_in_dim(
            pad_arr[i], start_index=i, slice_size=win_length
        )

    num_pad = win_length - 1
    # lpad with zeros so we can grab constant size windows even for rows 0..<win_length
    padded_arr = jnp.pad(
        arr, ((0, 0), (num_pad, 0)), mode="constant", constant_values=0
    )
    rows = jnp.arange(arr.shape[0])
    return jax.vmap(grab_for_row, in_axes=(None, 0))(padded_arr, rows)


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints/transformer_model_5000_samples").absolute()
    model: transformerlib.TransformerLM = utils.load_model(
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

    p_stay = 0.95
    p_switch = 1 - p_stay
    states, emissions, optimal_posterior = utils.sample_rollouts(
        n=1,
        sequence_length=512,
        initial_probs=jnp.ones((2,)) / 2,
        transition_matrix=jnp.array(
            [
                [p_stay, p_switch],
                [p_switch, p_stay],
            ]
        ),
        emission_matrix=jnp.array([[1 / 6] * 6, [1 / 10] * 5 + [1 / 2]]),
        rng=jax.random.key(0),
    )

    emissions = emissions.squeeze(-1)  # transformer expects (n, seq_len) input
    optimal_posterior = optimal_posterior.squeeze(
        -2
    )  # Produces (n, seq_len, num_classes)

    transformer_posterior = jax.nn.softmax(model(emissions), axis=-1)
    # First layer attn weights. [0] at the end is because we only called the function once
    # Shape (batch, num_heads, seq_len(q), seq_len(kv))
    # Entry ...,i,j is the degree to which token 'i' attends to token 'j'
    attn_weights = model.transformer.layers[0].attn_weights[0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["white", "red"], N=256
    )

    state_changes = jnp.flatnonzero(jnp.diff(states[0, :]) != 0)
    # Idx 'i' in state_changes means the state changes between 'i' and 'i+1'

    fig, ax = plt.subplots()
    ax.set_title("Posterior P('6')")
    ax.plot(transformer_posterior[0, :, -1], label="Transformer")
    ax.plot(optimal_posterior[0, :, -1], label="Optimal")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("P('6')")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(attn_weights[0, 0], cmap=cmap)
    for state_change_idx in state_changes:
        ax.axvline(state_change_idx + 1, color="k", linestyle="--", alpha=0.5)
        ax.axhline(state_change_idx + 1, color="k", linestyle="--", alpha=0.5)
    im.set_clim(0, 1)
    ax.set_title("Attention Weights")
    ax.set_ylabel("Query Tokens")
    ax.set_xlabel("Key/Value Tokens")
    ax.set_xticks(range(0, 512, 64))
    ax.set_yticks(range(0, 512, 64))
    ax.set_xticklabels(range(0, 512, 64))
    ax.set_yticklabels(range(0, 512, 64))
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")
    cbar.set_ticks([0, 0.5, 1])
    plt.show()

    windowed = grab_diag_window(attn_weights[0, 0], 25)
    means = jnp.mean(windowed, axis=0)
    fig, ax = plt.subplots()
    ax.set_title("Mean attendance to nearby tokens")
    x_ax = -jnp.arange(len(means))
    x_ax = jnp.flip(x_ax)
    ax.plot(x_ax, means, color="k", label="Mean (all)")
    # segments = jnp.split(windowed, 16, axis=0)
    # segments = jnp.stack(segments, axis=0).mean(axis=1)  # (n_segments, win_length)
    # colors = mpl.cm.viridis(np.linspace(0, 1, segments.shape[0]))
    # for i, seg in enumerate(segments):
    #     ax.plot(x_ax, seg, color=colors[i], alpha=0.5)
    # for state_id in jnp.unique(states[0, :]):
    #     state_indices = states[0, :] == state_id
    #     mean_for_state = windowed[state_indices, :].mean(axis=0)
    #     ax.plot(x_ax, mean_for_state, label=f"Hidden State={int(state_id)}")
    # state_change_surround = jnp.zeros(states.shape[1], dtype=bool)
    # for state_change_idx in state_changes:
    #     start = jnp.maximum(0, state_change_idx - 12)
    #     end = jnp.minimum(states.shape[1], state_change_idx + 12)
    #     state_change_surround = state_change_surround.at[start:end].set(True)

    near_change_mean = windowed[state_change_surround, :].mean(axis=0)
    away_change_mean = windowed[~state_change_surround, :].mean(axis=0)
    ax.plot(x_ax, near_change_mean, label="Mean (near state change)", color="r")
    ax.plot(x_ax, away_change_mean, label="Mean (others)", color="b")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative time")
    ax.set_ylabel("Attention Weight")
    ax.grid()
    plt.show()
    # breakpoint()
