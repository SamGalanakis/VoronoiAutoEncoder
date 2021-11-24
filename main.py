from typing import ForwardRef
import torch
import matplotlib.pyplot as plt
from torch.nn import parameter
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ImageDataset
from torch import nn
import wandb
import einops
import math
import torch.nn.functional as F
from torch import einsum

def main():
    run = wandb.init(project="GeomAutoencoder", entity="samme013",
                     config="configs/config.yaml", mode="online")

    config = run.config

    device = "cuda"

    transformations = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )



    dataset = ImageDataset("data\celeba_hq_256", transformations)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, num_workers=8, pin_memory=True,drop_last=True)

    parameters = []

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=True)
    

   

    class VoronoiNet(nn.Module):
        def __init__(self,patch_embed_dim=64,inner_dim=32):
            super().__init__()
            self.patch_embed_dim = patch_embed_dim
            self.inner_dim = inner_dim 
            self.conv_layers = nn.Sequential(torch.nn.Conv2d(3, 100,16,16,0,bias=False))

            self.patch_mapper = nn.Linear(768,64)
            self.coordinate_to_key= nn.Sequential(nn.Linear(2,inner_dim,bias=False),nn.ReLU(),nn.Linear(inner_dim,inner_dim,bias=False))
            self.positional_encoding = nn.Parameter(torch.randn((1,729,768),requires_grad=True))
            self.patch_to_key_val =  nn.Linear(patch_embed_dim, inner_dim * 2, bias=False)
            self.color_mapper = nn.Sequential( nn.Linear(inner_dim,inner_dim),nn.GELU(),nn.Linear(inner_dim,256),nn.GELU(),nn.Linear(256,64), nn.GELU(), nn.Linear(64,3))     
        def forward(self,x,coordinates):
            patches = x.unfold(2, 16, 8).unfold(3, 16, 8)
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


    model = VoronoiNet().to(device)

    if config.checkpoint_path != "":
        state_dict = torch.load(config.checkpoint_path)

        model.load_state_dict(state_dict)
    model = model.to(device)

    
    
    
    parameters += model.parameters()

    optimizer = torch.optim.Adam(parameters, lr=config['lr'])

    indexes = (torch.cartesian_prod(torch.arange(0, config.resize_size), torch.arange(
        0, config.resize_size))/(config.resize_size-1)).to(device)

    last_indexer = torch.arange(3).to(device)
    first_indexer = torch.arange(
        config.batch_size).reshape((-1, 1, 1)).to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.5,patience = len(dataloader))
    
    criterion = nn.L1Loss()



    
    
    for epoch in range(config.n_epochs):
        epoch_loss = 0

        for index, img in enumerate(tqdm(dataloader)):
            img = img.to(device)

            coordinates = torch.rand((config.batch_size,config.n_vertices,2),device=device)
            colors = model(img,coordinates)

            
            closest = torch.linalg.norm(
                (coordinates.unsqueeze(2) - indexes), dim=-1).argmin(axis=1)
            out_img = einops.rearrange(colors[first_indexer, closest.unsqueeze(
                -1), last_indexer], "b (i k) n -> b n i k ", i=config.resize_size, k=config.resize_size)

            l1_loss = criterion(out_img, img)

            loss =  l1_loss  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                if index % 100 == 0:
                    original_img = wandb.Image(inv_normalize(img[0].cpu()))
                    recreated_img = wandb.Image(
                        inv_normalize(out_img[0].cpu()))

                    wandb.log({
                        "Original": original_img,
                        "Recreated": recreated_img
                    })
            wandb.log({"Loss": loss.item(), "L1 loss": l1_loss.item(
            ),"lr": optimizer.param_groups[0]['lr']})
            epoch_loss = epoch_loss * (index)/(index+1) + loss.item()/(index+1)

        wandb.log({"epoch_loss": epoch_loss})


if __name__ == '__main__':
    main()
