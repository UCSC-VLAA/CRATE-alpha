# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities shared across models."""

from absl import logging
import helpers.utils as u
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Any
# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention core modules for Flax."""

import functools
from typing import (Any, Callable, Optional, Tuple)
from flax.linen.dtypes import promote_dtype

from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from einops import rearrange, repeat

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

# from .crate import posemb_sincos_2d

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32, cls_token=False):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  if cls_token:
      pe = jnp.concatenate([jnp.zeros([1, width]), pe], axis=0)
  return jnp.asarray(pe, dtype)[None, :, :]



def dot_product_attention_weights(query: Array,
                                  key: Array,
                                  bias: Optional[Array] = None,
                                  mask: Optional[Array] = None,
                                  broadcast_dropout: bool = True,
                                  dropout_rng: Optional[PRNGKey] = None,
                                  dropout_rate: float = 0.,
                                  deterministic: bool = False,
                                  dtype: Optional[Dtype] = None,
                                  precision: PrecisionLike = None):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype(query, key, dtype=dtype)
  dtype = query.dtype

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                            precision=precision)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  return attn_weights


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          broadcast_dropout: bool = True,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: Optional[Dtype] = None,
                          precision: PrecisionLike = None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype

  # compute attention weights


  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                            precision=precision)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # return weighted sum over values for each query position
  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value,
                    precision=precision)


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  qkv_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  out_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention
  decode: bool = False
  qkv_dot_general: DotGeneralT = lax.dot_general
  out_dot_general: DotGeneralT = lax.dot_general

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        f'Memory dimension ({qkv_features}) must be divisible by number of'
        f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads
    
    dense = functools.partial(
        DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.qkv_kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        dot_general=self.qkv_dot_general,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    dropout_rng = None
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param('deterministic', self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.out_kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        dot_general=self.out_dot_general,
        name='out', # type: ignore[call-arg]
    )(x)
    return out


class MultiHeadDotProductSubspaceAttention(Module):
    """Multi-head dot-product subspace attention.

      Attributes:
        num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
          should be divisible by the number of heads.
        dtype: the dtype of the computation
          (default: infer from inputs and params)
        param_dtype: the dtype passed to parameter initializers (default: float32)
        qkv_features: dimension of the key, query, and value.
        out_features: dimension of the last projection
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rate: dropout rate
        deterministic: if false, the attention weight is masked randomly
          using dropout, whereas if true, the attention weights
          are deterministic.
        precision: numerical precision of the computation see `jax.lax.Precision`
          for details.
        kernel_init: initializer for the kernel of the Dense layers.
        bias_init: initializer for the bias of the Dense layers.
        use_bias: bool: whether pointwise QKVO dense transforms use bias.
        attention_fn: dot_product_attention or compatible function. Accepts
          query, key, value, and returns output of shape
          `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
        decode: whether to prepare and use an autoregressive cache.
    """
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    qkv_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    out_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @compact
    def __call__(self,
                 inputs: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product subspace attention on the input data.

        Projects the inputs into multi-headed projection vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs: input signal of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features =  inputs.shape[-1]
        qkv_features = inputs.shape[-1]
       
        head_dim = qkv_features // self.num_heads

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]

        # proj = DenseGeneral(name='proj',
        #                     axis=-1,
        #                     # dtype=self.dtype,
        #                     features=(self.num_heads, head_dim),
        #                     use_bias=False,
        #                     # dot_general=self.qkv_dot_general
        # )(inputs)

        proj = DenseGeneral(name='proj',
                            axis=-1,
                            # dtype=self.dtype,
                            features=features,
                            use_bias=False,
                            # dot_general=self.qkv_dot_general
        )(inputs)
        
        proj =  rearrange(proj, 'b n (h d) -> b h n d', h = self.num_heads)

        query,key,value = proj, proj, proj
    
        # calculate attention matrix
        depth = query.shape[-1] # 64
        # print(f'depth:',depth)
        query = query / jnp.sqrt(depth)
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

        # normalize the attention weights
        attn_weights = jax.nn.softmax(attn_weights)

        # return weighted sum over values for each query position
        x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

        x = rearrange(x, 'b n h d -> b n (h d)')
        
        out = DenseGeneral(
            features=features,
            axis=-1,
            use_bias=False,
            # dtype=self.dtype,
            # dot_general=self.out_dot_general,
            name='out', # type: ignore[call-arg]
        )(x)
        return out


def merge_params(loaded, inited, load_clip=True, dont_load=()):
  """Makes `loaded` pytree match `init`, warning or failing on mismatch.

  Args:
    loaded: pytree of parameters, typically loaded from a checkpoint.
    inited: pytree of parameter, typically coming from model init.
    dont_load: List of regexes for parameters which shall not be taken
      from `loaded`, either because they should remain at their init value,
      or because they are missing on either side.

  Returns:
    If successful, a new pytree which matches the structure of `init`
    but contains values from `loaded`, except for `dont_load`.

    If structures don't match and mismatches are not covered by regexes in
    `dont_load` argument, then raises an exception with more information.
  """
  if inited is None:  # A useful shortcut for example for colabs.
    return loaded

  dont_load = u.check_and_compile_patterns(dont_load)

  def should_merge(name):
    return not any(pattern.fullmatch(name) for pattern in dont_load)

  loaded_flat, _ = u.tree_flatten_with_names(loaded)
  inited_flat, _ = u.tree_flatten_with_names(inited)
  new_loaded_flat = {}

  for k, v in loaded_flat:
    if 'img/' in k and load_clip:
        k = k.replace('img/', '')
    new_loaded_flat[k] = v
  loaded_flat = new_loaded_flat
  #loaded_flat = {k: v for k, v in loaded_flat}
  inited_flat = {k: v for k, v in inited_flat}

  # Let's first build the pytree from all common keys.
  merged = {}
  for name, init_val in inited_flat.items():
    # param is present in both. Load or ignore it!
    if name in loaded_flat and should_merge(name):
      merged[name] = loaded_flat[name]
    elif name == 'embedding/kernel':
      logging.info("resize %s patch emb %s to %s", name, loaded_flat[name].shape, inited_flat[name].shape)
      merged[name] = jax.image.resize(image=loaded_flat[name], shape=inited_flat[name].shape, method='bilinear')
    elif name == 'img/cls':
      logging.info("resize %s patch emb %s to %s", name, loaded_flat[name].shape, inited_flat[name].shape)
      merged[name] = jax.image.resize(image=loaded_flat[name], shape=inited_flat[name].shape, method='linear')
    elif name == 'img/embedding/kernel':
      logging.info("resize %s patch emb %s to %s in clip model", name, loaded_flat[name].shape, inited_flat[name].shape)
      merged[name] = jax.image.resize(image=loaded_flat[name], shape=inited_flat[name].shape, method='bilinear')
    elif name == 'img/pos_embedding':
        logging.info(
            'fixed pos_embedding cannot be stored, re-intialized needed')
        _, l, c = inited_flat["img/pos_embedding"].shape
        h, w = (l - 1) ** .5, (l - 1) ** .5
        if name in loaded_flat:
            logging.info('interplotate img pos embedding')
            merged[name] = jax.image.resize(loaded_flat[name], shape=inited_flat[name].shape, method='bilinear')
        else:
            merged[name] = posemb_sincos_2d(h, w, c, cls_token=True)
    elif name == 'txt/pos_embedding':
        logging.info(
            'txt pos_embedding cannot be stored, re-intialized needed')
        _, l, c = inited_flat[name].shape
        merged[name] = jax.image.resize(
            loaded_flat[name],
            shape=inited_flat[name].shape,
            method='bilinear')
    else:
      logging.info("Ignoring checkpoint and using init value for %s", name)
      merged[name] = init_val

  def pp(title, names, indent="  "):  # Just pretty-printing
    if names:
      return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))
    else:
      return ""

  # Now, if there are keys that only exist in inited or loaded, be helpful:
  not_in_loaded = inited_flat.keys() - loaded_flat.keys()
  not_in_inited = loaded_flat.keys() - inited_flat.keys()
  logging.info(pp("Parameters in model but not in checkpoint", not_in_loaded))
  logging.info(pp("Parameters in checkpoint but not in model", not_in_inited))

  # And now see if any of them are not explicitly ignored => an error
  not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
  not_in_inited = {k for k in not_in_inited if should_merge(k)}

  if not_in_loaded or not_in_inited:
    raise ValueError(
        pp("Params in checkpoint", loaded_flat.keys()) + "\n" +
        pp("Params in model (code)", inited_flat.keys()) + "\n" +
        pp("Params in model (code) but not in checkpoint and not `dont_load`ed",
           not_in_loaded, indent=" - ") + "\n" +  # Special indent for tests.
        pp("Params in checkpoint but not in model (code) and not `dont_load`ed",
           not_in_inited, indent=" + "))  # Special indent for tests.

  return u.recover_tree(merged.keys(), merged.values())


class AddPositionEmbs(nn.Module):
  """Adds positional embeddings to the inputs, supports caching for decode.

  Attributes:
    decode: whether to run in single-position autoregressive mode.
  """
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, posemb):
    """Applies AddPositionEmbs module.

    Adds posemb to the inputs, supports single-position autoregressive mode.

    Args:
      inputs: input data [batch_size, seq_len, emb_dim].
      posemb: positional embeddings.

    Returns:
      output: inputs modulated by pos-embeddings [batch_size, seq_len, emb_dim].
    """
    assert inputs.ndim == 3, f"Unexpected inputs shape: {inputs.shape}"
    _, seq_len, emb_dim = inputs.shape
    pe = posemb[:, :seq_len, :]

    if self.decode:
      is_initialized = self.has_variable("cache", "cache_index")
      # We use a cache position index for tracking decoding position.
      cache_index = self.variable("cache", "cache_index",
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        # Returns posemb[0, i, :], the positional embedding for the
        # current decoding position.
        pe = jax.lax.dynamic_slice(posemb,
                                   start_indices=jnp.array((0, i, 0)),
                                   slice_sizes=(1, 1, emb_dim))
    return inputs + pe


class DropPath(nn.Module):
    dropout_prob: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
      deterministic = nn.merge_param(
        "deterministic", self.deterministic, deterministic
      )
      if deterministic:
        return input
      keep_prob = 1 - self.dropout_prob
      shape = (input.shape[0],) + (1,) * (input.ndim - 1)
      rng = self.make_rng("drop_path")
      random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=jnp.float32)
      random_tensor = jnp.floor(random_tensor)
      return jnp.divide(input, keep_prob) * random_tensor