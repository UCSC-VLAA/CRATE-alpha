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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""
import jax
from typing import Optional, Sequence, Union

from absl import logging
from helpers import utils
from  models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.numpy as jnp
import numpy as np
import scipy.ndimage

from models.common import DropPath
from models.common import posemb_sincos_2d
from flax.linen.partitioning import remat


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32, cls_token=False):
  if typ == "learn":
    num_token = 1 if cls_token else 0
    return self.param(name,
                      nn.initializers.normal(stddev=0.02),
                      (1, np.prod(seqshape) + num_token, width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype, cls_token=cls_token)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")



class OvercompleteISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    eta: float = 0.1
    lmbda: float = 0.1
    dropout: float = 0.0
 
    @nn.compact
    def __call__(self, x, deterministic=True):
        """Applies CRATE OvercompleteISTABlock module."""
        n, l, d = x.shape  # pylint: disable=unused-variable
        D = self.param("D",
                       nn.initializers.kaiming_uniform(),
                       (d, 4 * d))
        
        
        D1 = self.param("D1",
                           nn.initializers.kaiming_uniform(),
                           (d, 4 * d))

       
        negative_lasso_grad = jnp.einsum("p d, n l p -> n l d", D, x)
        z1 = nn.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

       
        Dz1= jnp.einsum("d p, n l p -> n l d", D, z1)
        lasso_grad = jnp.einsum("p d, n l p -> n l d", D, Dz1 - x)
        z2 = nn.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        
        xhat = jnp.einsum("d p, n l p -> n l d", D1, z2)
     
        return xhat



class LayerScale(nn.Module):
    d:  int = 1024
    init_values: float = 1e-5
    name: str='layer_scale'

    @nn.compact
    def __call__(self, x):

        return x * self.param(
            self.name,
            nn.initializers.constant(self.init_values),
            (self.d),)



class Encoder1DBlock(nn.Module):
  """Single CRATE encoder block (MHSSA + ISTA)."""
  eta: float = 0.1
  lmbda: float = 0.1
  num_heads: int = 12
  dropout: float = 0.0
  drop_path: float = 0.0
  init_values: float = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    y = nn.LayerNorm()(x)

    ## hack init func
    y = out["sa"] = common.MultiHeadDotProductSubspaceAttention(
        num_heads=self.num_heads,
        qkv_kernel_init=nn.initializers.normal(stddev=0.02),
        out_kernel_init=nn.initializers.normal(stddev=0.02),
        bias_init=nn.initializers.zeros,
        deterministic=deterministic,
        use_bias=False
    )(y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    if self.init_values is not None:
        n, l, d = y.shape
        y = LayerScale(d, init_values=self.init_values, name='ls1')(y)
    y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)


    y = out["mlp"] = OvercompleteISTABlock(
            eta=self.eta, lmbda=self.lmbda, dropout=self.dropout
        )(y, deterministic)

    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    if self.init_values is not None:
        n, l, d = y.shape
        y = LayerScale(d, init_values=self.init_values, name='ls2')(y)
    y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
    
    x = out["+mlp"] = y + x


    return x, out


class Encoder(nn.Module):
  """CRATE Encoder for sequence to sequence translation."""
  depth: int
  eta: float = 0.1
  lmbda: float = 0.1
  num_heads: int = 12
  dropout: float = 0.0
  drop_path: float = 0.0
  init_values: float = None
  remat_policy: str = "none"


  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    dpr = [float(x) for x in np.linspace(0, self.drop_path, self.depth)] # drop path decay
    # Input Encoder
    BlockLayer = Encoder1DBlock

    if self.remat_policy == "minimal":
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    elif self.remat_policy == 'none':
        BlockLayer = Encoder1DBlock
    else:
        policy = None
        logging.info(f"activation checkpointing {self.remat_policy}")
        BlockLayer = remat(  # pylint: disable=invalid-name
            Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
        )  # "deterministic" is a static argu

    for lyr in range(self.depth):
      block = BlockLayer(
          name=f"encoderblock_{lyr}",
          eta=self.eta, lmbda=self.lmbda, num_heads=self.num_heads,
          dropout=self.dropout, drop_path=dpr[lyr],
          init_values=self.init_values
                    )
      x, out[f"block{lyr:02d}"] = block(x, deterministic)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return x, out



class _Model(nn.Module):
  """CRATE model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  eta: float = 0.1
  lmbda: float = 0.1
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  dropout: float = 0.0
  drop_path: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  patch_embeding: str = 'conv'
  init_values: float = None
  remat_policy: str = 'none'
  post_norm: bool = False
  emb_head_bias: bool = True
  mean: Sequence[float] = (0.485, 0.456, 0.406)
  std: Sequence[float] = (0.229, 0.224, 0.225)
  final_drop: float = 0.


  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    if self.post_norm:
        mean = jnp.asarray(
           self.mean)[None, None, None, :]
        std = jnp.asarray(
            self.std)[None, None, None, :]
        image = (image - mean) / std

   
    x = out["stem"] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        kernel_init=nn.initializers.kaiming_uniform(),
        use_bias=self.emb_head_bias,
        bias_init=nn.initializers.zeros,
        padding="VALID", name="embedding")(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    
  
    cls = self.param("cls", nn.initializers.normal(1e-6), (1, 1, c), x.dtype)
 
    x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)


    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype, cls_token=True)

   

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)
   
    x, out["encoder"] = Encoder(
        depth=self.depth,
        eta=self.eta,
        lmbda=self.lmbda,
        num_heads=self.num_heads,
        dropout=self.dropout,
        drop_path=self.drop_path,
        init_values = self.init_values,
        remat_policy=self.remat_policy,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    x =  nn.LayerNorm(name="encoder_norm")(x)
    x = out["head_input"] = x[:, 0]
    encoded = encoded[:, 1:]

    x_2d = jnp.reshape(encoded, [n, h, w, -1])



    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
        
      head = nn.Dense(self.num_classes, name="head",
                      kernel_init=nn.initializers.normal(stddev=0.02),
                      bias_init=nn.initializers.zeros,
                      use_bias=self.emb_head_bias,
                     )
      x = nn.Dropout(rate=self.final_drop)(x, not train)
   
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      "width": {"Ti": 192, "S": 576, "B": 768, "L": 1024, "H": 1280, "g": 1408, "G": 1664, "e": 1792}[v],
      "depth": {"Ti": 12, "S": 12, "B": 12, "L": 24, "H": 32, "g": 40, "G": 48, "e": 56}[v],
      "eta": {"Ti": 0.1, "S": 0.1, "B": 0.1, "L": 0.1, "H": 0.1, "g": 0.1, "G": 0.1, "e": 0.1}[v],
      "lmbda": {"Ti": 0.1, "S": 0.1, "B": 0.1, "L": 0.1, "H": 0.1, "g": 0.1, "G": 0.1, "e": 0.1}[v],
      "mlp_dim": {"Ti": 768, "S": 2304, "B": 3072, "L": 4096, "H": 5120, "g": 6144, "G": 8192, "e": 15360}[v],
      "num_heads": {"Ti": 3, "S": 12, "B": 12, "L": 16, "H": 16, "g": 16, "G": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  # extract cls
  cls_pos = old[:, 0:1, :]
  old = old[:, 1: , :]
  new = new[:, 1:, :]

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)

  #add cls
  grid = jnp.concatenate([cls_pos, grid], axis=1)
  return jnp.array(grid)


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Original ViT paper variant had posemb in a module:
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # Widely used version before 2022 had posemb in Encoder:
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # Old vit.py used to first concat [cls] token, then add posemb.
  # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
  # so we changed to add posemb then concat [cls]. We can recover the old
  # checkpoint by manually summing [cls] token and its posemb entry.
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP-head variants during ViT-G development had it inlined:
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  del model_cfg

  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(None, init_file)

  #restored_params = fix_old_checkpoints(restored_params)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load=dont_load)


  # resample posemb if needed.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  if 'pos_embedding'  in dont_load:
      logging.info('fixed pos_embedding cannot be stored, re-intialized needed')
      _, l, c = init_params["pos_embedding"].shape
      h, w = (l-1)**.5, (l-1)**.5
      #restored_params['pos_embedding'] = get_2d_sincos_pos_embed(c, h, cls_token=True)
      restored_params['pos_embedding'] = posemb_sincos_2d( h, w, c, cls_token=True)

  from helpers.utils import recover_dtype
  restored_params =  jax.tree_map(recover_dtype, restored_params)
  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",
    # pylint: disable=line-too-long
    # pylint: enable=line-too-long
}
