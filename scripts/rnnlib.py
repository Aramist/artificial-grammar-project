# == Define Model == #

# This is a model with a multiple linear RNNs stack on top of each other
# with nonlinear multi-layer perceptrons connecting them.

# Inspired by: https://arxiv.org/abs/2303.06349

import jax
from flax import nnx
from flax.nnx import Linear, Sequential, gelu
from flax.nnx.nn.recurrent import RNN, SimpleCell


class ResidualLayer(nnx.Module):
    def __init__(self, dimension: int, *, residual=True, rngs: nnx.Rngs):
        self.linear1 = Linear(dimension, dimension, rngs=rngs)
        self.bn = nnx.BatchNorm(dimension, rngs=rngs)
        self.residual = residual

    def __call__(self, x: jax.Array):
        return gelu(self.bn(self.linear1(x))) + x


class LRUBlock(nnx.Module):
    def __init__(self, dim_in, dim_out, num_layers, *, rngs: nnx.Rngs):
        self.linear_in = Linear(dim_in, dim_out, rngs=rngs)
        self.cell = SimpleCell(
            in_features=dim_in,
            hidden_features=dim_out,
            rngs=rngs,
            activation_fn=(lambda x: x),
        )
        self.rnn = RNN(self.cell, rngs=rngs)
        self.mlp_layers = [ResidualLayer(dim_out, rngs=rngs) for _ in range(num_layers)]
        self.mlp = Sequential(*self.mlp_layers)

    def __call__(self, x):
        return self.mlp(self.rnn(x))


class LRUModel(nnx.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        mlp_depth: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.mlp_depth = mlp_depth
        self.num_layers = num_layers
        self.rngs = rngs

        self.lru_blocks = [
            LRUBlock(dim_in=dim_in, dim_out=dim_hidden, num_layers=mlp_depth, rngs=rngs)
        ] + (num_layers - 1) * [
            LRUBlock(
                dim_in=dim_hidden, dim_out=dim_hidden, num_layers=mlp_depth, rngs=rngs
            )
        ]
        self.lrus = Sequential(*self.lru_blocks)
        self.readout_layer = Linear(dim_hidden, dim_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Evaluate the model on the input data.
        The input data is a batch of sequences of indices in the range [0, vocab_size).

        Args:
            x (jax.Array): The input data. The shape is (batch_size, seq_len). Dtype is int32.

        Returns:
            jax.Array: The output of the model. The shape is (batch_size, seq_len, vocab_size).
        """
        return self.readout_layer(self.lrus(nnx.one_hot(x, 6)))
