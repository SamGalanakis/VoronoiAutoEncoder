import torch
from torch import einsum, nn
import einops
    
    
class VoronoiNet(nn.Module):
    def __init__(self,patch_embed_dim=64,inner_dim=32,patch_size = 8):
        super().__init__()
        self.patch_embed_dim = patch_embed_dim
        self.inner_dim = inner_dim 
        self.patch_size = patch_size
        self.conv_layers = nn.Sequential(torch.nn.Conv2d(3, 100,16,16,0,bias=False))

        self.patch_mapper = nn.Linear(192,patch_embed_dim)
        self.coordinate_to_key= nn.Sequential(nn.Linear(2,inner_dim,bias=False),nn.ReLU(),nn.Linear(inner_dim,inner_dim,bias=False))
        self.positional_encoding = nn.Parameter(torch.randn((1,1369,3*(self.patch_size**2)),requires_grad=True))
        self.patch_to_key_val =  nn.Linear(patch_embed_dim, inner_dim * 2, bias=False)
        self.color_mapper = nn.Sequential( nn.Linear(inner_dim,inner_dim),nn.GELU(),nn.Linear(inner_dim,256),nn.GELU(),nn.Linear(256,64), nn.GELU(), nn.Linear(64,3))     
    def forward(self,x,coordinates):
        patches = x.unfold(2, self.patch_size, 6).unfold(3, self.patch_size, 6)
        patches = einops.rearrange(patches,"b c i j k m -> b (i j) (c k m)")
        patches = patches + self.positional_encoding
        patches = self.patch_mapper(patches)

        q = self.coordinate_to_key(coordinates)
        k,v = self.patch_to_key_val(patches).split((self.inner_dim,self.inner_dim),-1)
        sim = einsum('b i d, b j d -> b i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        colors = self.color_mapper(out)
        return colors  