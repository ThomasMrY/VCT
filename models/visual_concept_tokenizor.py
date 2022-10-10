from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
import gin
import numpy as np

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class MLP_head(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_cls)
        )
    def forward(self, x):
        return self.net(x)

# main class
@gin.configurable
class VCT_Encoder(nn.Module):
    def __init__(
        self,
        *,
        index_num = 10,
        depth=6,
        dim=256,
        z_index_dim = 10,
        latent_dim = 256,
        cross_heads = 1,
        latent_heads = 6,
        cross_dim_head = 128,
        latent_dim_head = 128,
        weight_tie_layers = False,
        ce_loss= False,
        max_freq = 10,
        num_freq_bands = 6,
        emb=False,
        fc = False,
        num_cls = False,
        emb_cls = False
    ):
        super().__init__()
        num_latents = z_index_dim
        self.components = z_index_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.depth = depth


        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim), True)
        self.cs_layers = nn.ModuleList([])
        for i in range(depth):
            self.cs_layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim + 26, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim + 26),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))

        if ce_loss:
            self.fc_layer = nn.Linear(dim, index_num)

        get_latent_attn = lambda: PreNorm(dim + 26, Attention(dim + 26, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(dim + 26, FeedForward(dim + 26))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth-1):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        if emb:
            self.emb = nn.Parameter(torch.randn(num_latents, latent_dim), True)
        self.fc = fc
        if fc:
            self.mlp_head = MLP_head(latent_dim, latent_dim//2, num_cls)
            if emb_cls:
                self.emb_cls = nn.Parameter(torch.randn(num_cls, latent_dim), True)

    def forward(
        self,
        data,
        mask = None
    ):
        b, *axis, device = *data.shape, data.device
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), (int(np.sqrt(axis[0])),int(np.sqrt(axis[0])))))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        data = torch.cat((data, enc_pos.reshape(b,-1,enc_pos.shape[-1])), dim = -1)
        x0 = repeat(self.latents, 'n d -> b n d', b = b)
        for i in range(self.depth):
            cross_attn, cross_ff = self.cs_layers[i]

            # cross attention only happens once for Perceiver IO

            x = cross_attn(x0, context = data, mask = mask) + x0
            x0 = cross_ff(x) + x

            if i != self.depth - 1:
                self_attn, self_ff = self.layers[i]
                x_d = self_attn(data) + data
                data = self_ff(x_d) + x_d

        return x0

def swish(x):
    return x * torch.sigmoid(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)




class MLP_layer(nn.Module):
    def __init__(self, z_dim = 512, latent_dim = 256):
        super(MLP_layer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    def forward(self,x):
        return self.net(x)

class MLP_layers(nn.Module):
    def __init__(self, z_dim = 512, latent_dim = 256, num_latents=16):
        super(MLP_layers, self).__init__()
        self.nets = nn.ModuleList([MLP_layer(z_dim=z_dim, latent_dim=latent_dim) for i in range(num_latents)])


    def forward(self,x):
        out = []
        for sub_net in self.nets:
            out.append(sub_net(x)[:,None,:])
        return torch.cat(out, dim=1)
        



# main class
@gin.configurable
class VCT_Decoder(nn.Module):
    def __init__(
        self,
        *,
        depth = 4,
        index_num = 10,
        dim=256,
        z_index_dim = 64,
        latent_dim = 256,
        cross_heads = 1,
        cross_dim_head = 128,
        latent_heads = 6,
        latent_dim_head = 128,
        ce_loss= False,
        fourier_encode_data = False,
        weight_tie_layers = False,
        max_freq = 10,
        num_freq_bands = 6
    ):
        super().__init__()
        num_latents = z_index_dim
        self.components = z_index_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim), True)

        self.depth = depth
        if depth != 0:
            get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = latent_heads, dim_head = latent_dim_head))
            get_latent_ff = lambda: PreNorm(dim, FeedForward(dim))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            self.slayers = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}

            for i in range(depth-1):
                self.slayers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

        self.cs_layers = nn.ModuleList([])
        for i in range(depth):
            self.cs_layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))
        self.ce_loss = ce_loss
        if ce_loss:
            self.fc_layer = nn.Linear(dim, index_num)
        
        if depth != 0:
            get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
            get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
            get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

            self.layers = nn.ModuleList([])
            cache_args = {'_cache': weight_tie_layers}

            for i in range(depth):
                self.layers.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))


    def forward(
        self,
        data,
        mask = None
    ):
        b, *axis, device = *data.shape, data.device
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), (int(np.sqrt(axis[0])),int(np.sqrt(axis[0])))))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos.reshape(b,-1,enc_pos.shape[-1])), dim = -1)

        x = repeat(self.latents, 'n d -> b n d', b = b)
        cp_vals = data
        for i in range(self.depth):

            cross_attn, cross_ff = self.cs_layers[i]
            x = cross_attn(x, context = cp_vals, mask = mask) + x
            x = cross_ff(x) + x


            self_attn, self_ff = self.layers[i]
            x = self_attn(x) + x
            x = self_ff(x) + x

            if i != self.depth - 1:
                self_attn, self_ff = self.slayers[i]
                cp_vals = self_attn(cp_vals) + cp_vals
                cp_vals = self_ff(cp_vals) + cp_vals

        if self.ce_loss:
            return self.fc_layer(x)
        else:
            return x

