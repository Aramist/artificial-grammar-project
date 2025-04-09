from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from flax import nnx


def make_causal_attn_mask(num_tokens: int) -> jax.Array:
    """Creates a causal attention mask.

    Args:
        num_tokens (int): Sequence length

    Returns:
        jax.Array: Causal attention mask of shape (num_tokens, num_tokens). A 1 (True) at M_i,j means that token i can attend to token j.
    """

    # Since tokens can't attend to future tokens, the upper triangle (j > i) will be 0
    # Since tokens can attend to themselves, the diagonal (j = i) will be 1
    mask = jnp.tril(
        jnp.ones((num_tokens, num_tokens), dtype=bool)
    )  # Has ones at attended tokens
    return mask


@nnx.jit
def multi_head_sdpa(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    mask: jax.Array,
    rngs: nnx.Rngs,
    dropout_p: float,
    mask_value: float,
) -> tuple[jax.Array, jax.Array]:
    """Multi head scaled dot product attention for batched input. Computes the attention weights as defined by Vaswani et al. (2017).
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.

    Differs from the pytorch implementation in that the number of heads in the query must be equal to that of the key and value sequences.


    Args:
        query (jax.Array): Query sequence of shape (*batch, target_seq_len, num_heads, key_dim)
        key (jax.Array): Key sequence of shape (*batch, source_seq_len, num_heads, key_dim)
        value (jax.Array): Value sequence of shape (*batch, source_seq_len, num_heads, value_dim)
        mask (jax.Array): Boolean attention mask. True at M_i,j means that token i (query) can attend to token j (kv).
        rng_key (jax.Array): Random key for dropout.
        dropout_p (float, optional): Dropout probability.
        mask_value (float, optional): Value to use for unattended tokens. Ideally -inf but this sometimes causes gradient issues.

    Returns:
        jax.Array: Attention weights of shape (*batch, num_target_heads, target_seq_len, source_seq_len)
        jax.Array: Output of shape (*batch,target_seq_len, num_heads, value_dim)
    """

    # Turn the mask into a float mask
    attn_mask = (~mask).astype(jnp.float32) * mask_value

    # H: heads, Q: query (target) seq len, D: qk dim
    # K: kv (source) seq len
    numerator = jnp.einsum(
        "...QHD,...KHD->...HQK", query, key
    )  # out: (*batch, n_head, q_seq_len, kv_seq_len)
    scale = jnp.pow(query.shape[-1], -0.5)  # 1/sqrt(qk_dim), scalar

    # Apply dropout to the attention weights
    dropout_key = rngs.dropout()
    dropout_mask = ~jax.random.bernoulli(
        dropout_key, p=dropout_p, shape=numerator.shape
    )  # Has value 0 with probability dropout_p
    dropout_mask = dropout_mask.astype(jnp.float32)  # Convert to float
    attn_mask = jnp.broadcast_to(attn_mask, numerator.shape)

    attn_weights = dropout_mask * numerator * scale
    attn_weights = nnx.softmax(
        dropout_mask * numerator * scale + attn_mask, axis=-1
    )  # produces (*batch, n_head, q_seq_len, kv_seq_len)

    preproj_output = jnp.einsum(
        "...HQK,...KHD->...QHD", attn_weights, value
    )  # produces (*batch, q_seq_len, n_head, v_dim)
    # Produces (*batch, q_seq_len, n_head, v_dim)

    return attn_weights, preproj_output


class MultiheadAttention(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        *,
        dropout_p: float = 0.0,
        use_bias=True,
        rngs: nnx.Rngs,
    ):
        """Applies multi-head attention to transform an input sequence of shape
        (*batch, seq_len, num_heads * head_dim) into an output sequence of shape
        (*batch, seq_len, num_heads * head_dim). Via a key-value sequence of shape
        (*batch, kv_seq_len, num_heads * head_dim).

        Args:
            head_dim (int): Dimension of each attention head.
            num_heads (int): Number of attention heads.
            dropout_p (float, optional): Probability of each attention weight
                being dropped out. Defaults to 0.0.
            use_bias (bool, optional): Whether to use bias in the linear layers. Defaults to True.
            rng_key (Optional[jax.Array], optional): Random key for dropout. If None, a new key will be generated. Defaults to None.
        """
        super().__init__()
        self.num_features = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.use_bias = use_bias
        self.rngs = rngs

        # Create linear layers for q, k, and v
        self.q_proj = nnx.LinearGeneral(
            in_features=self.num_features,
            out_features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=use_bias,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=self.rngs,
        )
        self.k_proj = nnx.LinearGeneral(
            in_features=self.num_features,
            out_features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=use_bias,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=self.rngs,
        )
        self.v_proj = nnx.LinearGeneral(
            in_features=self.num_features,
            out_features=(self.num_heads, self.head_dim),
            axis=-1,
            use_bias=use_bias,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=self.rngs,
        )
        self.o_proj = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.num_features,
            axis=(-2, -1),
            use_bias=use_bias,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=self.rngs,
        )

    def forward(
        self,
        query_sequence: jax.Array,
        kv_sequence: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Applies multi-head attention to the input sequence via the
        scaled dot-product attention mechanism. For self-attention, the
        query and key-value sequences are the same.

        Args:
            query_sequence (jax.Array): Query sequence of shape (*batch, seq_len, num_heads * head_dim)
            kv_sequence (jax.Array): Key-value sequence of shape (*batch, kv_seq_len, num_heads * head_dim)
            mask (jax.Array): Boolean attention mask. True at M_i,j means that token i (query) can attend to token j (kv).
            rng_key (jax.Array): Random key for dropout.

        Returns:
            jax.Array: Output sequence of shape (*batch, seq_len, num_heads * head_dim)
            jax.Array: Attention weights of shape (*batch, num_heads, seq_len, kv_seq_len)
        """

        Q, K, V = (
            self.q_proj(query_sequence),
            self.k_proj(kv_sequence),
            self.v_proj(kv_sequence),
        )
        # New shape is (*batch, num_heads, seq_len, head_dim)

        # Apply attention
        attn_weights, output = multi_head_sdpa(
            Q,
            K,
            V,
            mask=mask,
            rngs=self.rngs,
            dropout_p=self.dropout_p,
            mask_value=float("-inf"),
        )

        # Merge heads
        output = self.o_proj(output)
        # New shape is (*batch, seq_len, num_heads * head_dim)

        return output, attn_weights

    def __call__(
        self, query_sequence: jax.Array, kv_sequence: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """Applies multi-head attention to the input sequence via the
        scaled dot-product attention mechanism. For self-attention, the
        query and key-value sequences are the same.

        Args:
            query_sequence (jax.Array): Query sequence of shape (*batch, seq_len, num_heads * head_dim)
            kv_sequence (jax.Array): Key-value sequence of shape (*batch, kv_seq_len, num_heads * head_dim)
            mask (jax.Array): Boolean attention mask. True at M_i,j means that token i (query) can attend to token j (kv).

        Returns:
            jax.Array: Output sequence of shape (*batch, seq_len, num_heads * head_dim)
            jax.Array: Attention weights of shape (*batch, num_heads, seq_len, kv_seq_len)
        """
        return self.forward(query_sequence, kv_sequence, mask)


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        *,
        attn_dropout_p: float = 0.0,
        rngs: nnx.Rngs,
    ):
        """Transformer layer with multi-head attention implemented according to Vaswani et al. (2017).

        Args:
            d_model (int): Dimension of input and embedding
            d_feedforward (int): Dimension of the internal feedforward layer
            attention_dropout_p (float, optional): Dropout probability for attention. Defaults to 0.0.
        """

        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p
        self.rngs = rngs

        if d_model % num_heads != 0:
            raise ValueError(
                f"Number of heads ({num_heads}) must divide model dimension ({d_model})"
            )
        self.head_dim = d_model // num_heads

        self.mha = MultiheadAttention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout_p=self.attn_dropout_p,
            rngs=self.rngs,
        )
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=self.rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(
                in_features=d_model,
                out_features=d_feedforward,
                kernel_init=nnx.initializers.xavier_uniform(),
                rngs=self.rngs,
            ),
            nnx.relu,
            nnx.Linear(
                in_features=d_feedforward,
                out_features=d_model,
                kernel_init=nnx.initializers.xavier_uniform(),
                rngs=self.rngs,
            ),
        )
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=self.rngs)

    def forward(self, sequence: jax.Array, mask: jax.Array) -> jax.Array:
        """Applies the transformer layer to the input sequence.

        Args:
            sequence (jax.Array): Input sequence of shape (*batch, seq_len, d_model)
            mask (jax.Array): Boolean attention mask. True at M_i,j means that token i (query) can attend to token j (kv).

        Returns:
            jax.Array: Output sequence of shape (*batch, seq_len, d_model)
        """
        mha_output, attn_weights = self.mha(
            query_sequence=sequence, kv_sequence=sequence, mask=mask
        )

        self.sow(nnx.Intermediate, "attn_weights", attn_weights)

        sequence = self.norm1(sequence + mha_output)

        sequence = self.norm2(sequence + self.ffn(sequence))

        return sequence

    def __call__(self, x: jax.Array, use_causal_mask: bool) -> jax.Array:
        """Applies the transformer layer to the input sequence.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len, d_model)
            use_causal_mask (bool): Whether to use a causal mask. If True, the mask will be upper triangular. If False, no mask will be applied.
        """
        # Create a mask if needed
        if use_causal_mask:
            mask = make_causal_attn_mask(x.shape[-2])
        else:
            mask = jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool)

        return self.forward(x, mask)


class Transformer(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        *,
        attn_dropout_p: float = 0.0,
        rngs: nnx.Rngs,
    ):
        """Transformer model with multiple layers of multi-head attention and feedforward networks.

        Args:
            num_layers (int): Number of transformer layers.
            d_model (int): Dimension of input and embedding.
            num_heads (int): Number of attention heads.
            d_feedforward (int): Dimension of the internal feedforward layer.
            attn_dropout_p (float, optional): Dropout probability for attention. Defaults to 0.0.
            rngs (nnx.Rngs): Random streams for parameter initialization and dropout.
        """
        self.layers = [
            TransformerLayer(
                d_model,
                num_heads,
                d_feedforward,
                attn_dropout_p=attn_dropout_p,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

    def forward(self, sequence: jax.Array, use_causal_mask: bool) -> jax.Array:
        """Applies the transformer model to the input sequence.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len, d_model)
            use_causal_mask (bool): Whether to use a causal mask. If True, the mask will be upper triangular. If False, no mask will be applied.
        """
        for layer in self.layers:
            sequence = layer(sequence, use_causal_mask=use_causal_mask)
        return sequence

    def __call__(self, x: jax.Array, use_causal_mask: bool) -> jax.Array:
        """Applies the transformer model to the input sequence.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len, d_model)
            use_causal_mask (bool): Whether to use a causal mask. If True, the mask will be upper triangular. If False, no mask will be applied.
        """
        return self.forward(x, use_causal_mask=use_causal_mask)


# Implement all the scaffolding
class Buffer(nnx.Variable):
    """Represents an array stored by a module, but not updated by the optimizer."""

    pass


class SinCosPositionalEmbedding(nnx.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        """Creates a sinusoidal positional embedding according to Vaswani et al. (2017).

        Args:
            d_model (int): Embedding dimensionality
            max_len (int, optional): Maximum sequence length. Defaults to 512.
        """

        self.d_model = d_model
        self.max_len = max_len

        # Create the positional embedding matrix
        position = jnp.arange(max_len)[:, None]
        # This part is 10000 ^ (2i / d_model)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
        pos_emb = jnp.zeros((max_len, d_model))
        pos_emb = pos_emb.at[:, 0::2].set(jnp.sin(position * div_term))
        pos_emb = pos_emb.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pos_emb = Buffer(pos_emb)
        # The positional embedding matrix is of shape (max_len, d_model)

    def forward(self, x: jax.Array) -> jax.Array:
        """Applies the positional embedding to the input sequence.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len, d_model)

        Returns:
            jax.Array: Positional embedding of shape (*batch, seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        pos_emb = jnp.broadcast_to(
            self.pos_emb[:seq_len, :], (batch_shape + (seq_len, self.d_model))
        )
        return x + pos_emb

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.forward(x)


class TransformerLM(nnx.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_embedding: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        attn_dropout_p: float = 0.0,
        rngs: nnx.Rngs,
    ):
        """Creates a transformer for language modeling

        Args:
            vocab_size (int): Number of tokens in the vocabulary
            max_seq_len (int): Maximum sequence length
            d_embedding (int): Dimension of the language embedding
            num_layers (int): Number of transformer layers
            d_model (int): Dimension of the transformer's hidden state
            num_heads (int): Number of attention heads
            d_feedforward (int): Dimension of the feedforward layers inside the transformer
            rngs (nnx.Rngs): Random streams for parameter initialization and dropout.
            attn_dropout_p (float, optional): Dropout probability for attention. Defaults to 0.0.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_embedding = d_embedding
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.attn_dropout_p = attn_dropout_p
        self.rngs = rngs
        self.lang_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_embedding,
            embedding_init=nnx.initializers.xavier_uniform(),
            rngs=self.rngs,
        )
        self.initial_linear = nnx.Linear(
            in_features=d_embedding,
            out_features=d_model,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=self.rngs,
        )
        self.final_linear = nnx.Linear(
            in_features=d_model,
            out_features=vocab_size,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=self.rngs,
        )

        self.pos_emb = SinCosPositionalEmbedding(d_model=d_model, max_len=max_seq_len)
        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            attn_dropout_p=attn_dropout_p,
            rngs=rngs,
        )

    def forward(self, x: jax.Array) -> jax.Array:
        """Applies the transformer to the input sequence for training.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len). Entries should be indices into
                the vocabulary.

        Returns:
            jax.Array: Output sequence of shape (*batch, seq_len, vocab_size) representing the logits for each token in the vocabulary.
        """
        # Apply the language embedding
        x = self.lang_embedding(x)  # (*batch, seq_len, d_embedding)
        # Apply the initial linear layer
        x = self.initial_linear(x)  # (*batch, seq_len, d_model)
        # Apply the positional embedding
        x = self.pos_emb(x)

        # Apply the transformer
        x = self.transformer(x, use_causal_mask=True)  # (*batch, seq_len, d_model)
        # Apply the final linear layer
        x = self.final_linear(x)  # (*batch, seq_len, vocab_size)
        return x

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the transformer to the input sequence for inference.

        Args:
            x (jax.Array): Input sequence of shape (*batch, seq_len). Entries should be indices into
                the vocabulary.

        Returns:
            jax.Array: Output sequence of shape (*batch, seq_len, vocab_size) representing the logits for each token in the vocabulary.
        """
        return self.forward(x)


############## PyTorch Implementation ##############


class PytorchSinCosPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        """Creates a sinusoidal positional embedding according to Vaswani et al. (2017).

        Args:
            d_model (int): Embedding dimensionality
            max_len (int, optional): Maximum sequence length. Defaults to 512.
        """

        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create the positional embedding matrix
        position = torch.arange(max_len)[:, None]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pos_emb = torch.zeros((max_len, d_model))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb)
        # The positional embedding matrix is of shape (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the positional embedding to the input sequence.

        Args:
            x (torch.Tensor): Input sequence of shape (*batch, seq_len, d_model)

        Returns:
            torch.Tensor: Positional embedding of shape (*batch, seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        pos_emb = self.pos_emb[:seq_len, :].expand(*batch_shape, seq_len, self.d_model)
        return x + pos_emb


class PytorchTransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_embedding: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        attn_dropout_p: float = 0.0,
    ):
        """Pytorch implementation of a transformer for language modeling for comparison with JAX.

        Args:
            vocab_size (int): Number of tokens in the vocabulary
            max_seq_len (int): Maximum sequence length
            d_embedding (int): Dimension of the language embedding
            num_layers (int): Number of transformer layers
            d_model (int): Dimension of the transformer's hidden state
            num_heads (int): Number of attention heads
            d_feedforward (int): Dimension of the feedforward layers inside the transformer
            attn_dropout_p (float, optional): Dropout probability for attention. Defaults to 0.0.
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_embedding = d_embedding
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.attn_dropout_p = attn_dropout_p

        self.lang_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_embedding,
        )
        self.initial_linear = torch.nn.Linear(
            in_features=d_embedding,
            out_features=d_model,
        )
        self.final_linear = torch.nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
        )
        self.pos_emb = PytorchSinCosPositionalEmbedding(
            d_model=d_model, max_len=max_seq_len
        )
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_feedforward,
                dropout=attn_dropout_p,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the transformer to the input sequence for training.

        Args:
            x (torch.Tensor): Input sequence of shape (*batch, seq_len). Entries should be indices into
                the vocabulary.

        Returns:
            torch.Tensor: Output sequence of shape (*batch, seq_len, vocab_size) representing the logits for each token in the vocabulary.
        """
        # Apply the language embedding
        x = self.lang_embedding(x)  # (*batch, seq_len, d_embedding)
        # Apply the initial linear layer
        x = self.initial_linear(x)  # (*batch, seq_len, d_model)
        # Apply the positional embedding
        x = self.pos_emb(x)  # (*batch, seq_len, d_model)
        # Apply the transformer
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            sz=x.shape[-2], device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=mask, is_causal=True)  # (*batch, seq_len, d_model)
        # (*batch, seq_len, d_model)
        # Apply the final linear layer
        x = self.final_linear(x)  # (*batch, seq_len, vocab_size)
        # At each position 't', the result is a distribution of logits for the token 't+1'
        return x


class PytorchHMMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        emissions: torch.Tensor,
        states: torch.Tensor,
    ):
        """Dataset for HMM training.

        Args:
            emissions (torch.Tensor): Emissions of shape (num_samples, seq_len, num_states)
            states (torch.Tensor): States of shape (num_samples, seq_len)
        """
        self.emissions = emissions
        self.states = states

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the emissions and states for a given index.

        Args:
            idx (int): Index of the sample to get

        Returns:
            torch.Tensor: Emissions of shape (seq_len)
            torch.Tensor: States int tensor of shape (seq_len)
        """
        return self.emissions[idx], self.states[idx]

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self.emissions.shape[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    norms = []
    lens = 2 ** jnp.arange(4, 12)
    for len in lens:
        test = SinCosPositionalEmbedding(int(len), 512)
        norm = jnp.linalg.norm(test.pos_emb) / jnp.sqrt(len)
        norms.append(norm.item())

    plt.plot(np.array(lens), norms)
    plt.show()
