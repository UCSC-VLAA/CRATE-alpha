import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class OvercompleteISTABlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, d, overcomplete_ratio=4, eta=0.1, lmbda=0.1):
        super(OvercompleteISTABlock, self).__init__()
        self.eta = eta
        self.lmbda = lmbda
        self.overcomplete_ratio = overcomplete_ratio
        self.d = d

        # Define the matrix D
        self.D = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))
        self.D1 = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d, overcomplete_ratio * d)))


    def forward(self, x):
        """Applies CRATE OvercompleteISTABlock module."""
        
        negative_lasso_grad = torch.einsum("pd,nlp->nld", self.D, x)
        z1 = F.relu(self.eta * negative_lasso_grad - self.eta * self.lmbda)

        Dz1 = torch.einsum("dp,nlp->nld", self.D, z1)
        lasso_grad = torch.einsum("pd,nlp->nld", self.D, Dz1 - x)
        z2 = F.relu(z1 - self.eta * lasso_grad - self.eta * self.lmbda)

        xhat = torch.einsum("dp,nlp->nld", self.D1, z2)
      
        return xhat
    

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, OvercompleteISTABlock(d = dim))
            ]))

    def forward(self, x):
      
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x 

class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size, bias=True, padding='valid')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):

        x = self.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

      
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
      
        x = self.transformer(x)

        x = x[:, 0]

        x = self.to_latent(x)
     
        return self.mlp_head(x)


def CRATE_base():
    return CRATE(image_size=224,
                patch_size=16,
                num_classes=1000,
                dim=768,
                depth=12,
                heads=12,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=768//12)

def CRATE_large():
    return CRATE(image_size=224,
                patch_size=14,
                num_classes=1000,
                dim=1024,
                depth=24,
                heads=16,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=1024//16)
