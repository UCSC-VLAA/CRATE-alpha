### CLIP source code from OpenAI:
# https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.normalization import _shape_t
from open_clip import get_tokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm."""

    def __init__(self, normalized_shape: _shape_t, eps: float = 0.000001, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class OvercompleteISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, d, overcomplete_ratio=4, eta=0.1, lmbda=0.1, decouple=True):
        super(OvercompleteISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.overcomplete_ratio = overcomplete_ratio
        self.decouple = decouple
        self.d = d

        # Define the matrix D
        self.c_fc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))
        self.c_proj = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))

    def forward(self, x):
        """Applies CRATE OvercompleteISTABlock module."""
        
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.c_fc, x)
        z1 = F.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

        Dz1 = torch.einsum("dp,nlp->nld", self.c_fc, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.c_fc, Dz1 - x)
        z2 = F.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        xhat = torch.einsum("dp,nlp->nld", self.c_proj, z2)
      
        return xhat
    

class VisionResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head,bias=False)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = OvercompleteISTABlock(d=d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
  
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        y = self.attention(self.ln_1(x))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU(approximate='tanh')),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class VTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[VisionResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn((input_resolution // patch_size) ** 2 + 1, width))

        self.transformer = VTransformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim))
    
    def forward(self, x: torch.Tensor):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]
        x = self.ln_post(x)
        x = x @ self.proj

        return x
    

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int, # 512
                 # vision
                 image_resolution: int, # 224
                 vision_layers: Union[Tuple[int, int, int, int], int], # 12
                 vision_width: int, # 768
                 vision_patch_size: int, # 16
                 vision_heads: int, #16
                 # text
                 context_length: int, # 77
                 vocab_size: int, # 49408
                 transformer_width: int, # 512
                 transformer_heads: int, # 8
                 transformer_layers: int # 12
                 ):
        super().__init__()
        self.context_length = context_length

    
        
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=None
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[:, -1, :]
        x = x @ self.text_projection.T
        return x

    
   
    

def get_crate_clipa():
    

    embed_dim = 1024 # vision and text dim 

    image_resolution = 224
    vision_patch_size = 14

    vision_width, vision_layers, vision_heads = get_vision_config('L').values() # crate_alpha_L14
    # vision_width, vision_layers, vision_heads = get_vision_config('H').values() # crate_alpha_H14

    vocab_size,context_length,transformer_width,transformer_layers = get_text_config('H').values()
    transformer_heads = transformer_width // 64
    
    
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_heads,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    
    model = model.eval()

    tokenizer =  get_tokenizer('hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B')
    return model, tokenizer



def get_text_config(model):
   return {
        'vocab_size': 32000,
        'context_length': 32,
        "width": { "B": 512, "L": 768, "H": 1024}[model],
        "depth": { "B": 12, "L": 12, "H": 24}[model],
    }

def get_vision_config(model):
 return {
      "width": { "B": 768, "L": 1024, "H": 1280}[model],
      "depth": { "B": 12, "L": 24, "H": 32}[model],
      "num_heads": {"B": 12, "L": 16, "H": 16}[model],
  }


