import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import ImageDataset
from torch import nn
import wandb
import einops
from models import VoronoiNet


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

    dataset = ImageDataset("data/", transformations)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, num_workers=8, pin_memory=True, drop_last=True)

    parameters = []

    model = VoronoiNet(patch_embed_dim=config.patch_embed_dim,
                       inner_dim=32).to(device)

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=int(len(dataloader)//3))

    criterion = nn.L1Loss()

    for epoch in range(config.n_epochs):
        epoch_loss = 0

        for index, img in enumerate(tqdm(dataloader)):
            img = img.to(device)

            coordinates = torch.rand(
                (config.batch_size, config.n_vertices, 2), device=device)
            colors = model(img, coordinates)

            closest = torch.linalg.norm(
                (coordinates.unsqueeze(2) - indexes), dim=-1).argmin(axis=1)
            out_img = einops.rearrange(colors[first_indexer, closest.unsqueeze(
                -1), last_indexer], "b (i k) n -> b n i k ", i=config.resize_size, k=config.resize_size)

            l1_loss = criterion(out_img, img)

            loss = l1_loss

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
            ), "lr": optimizer.param_groups[0]['lr']})
            epoch_loss = epoch_loss * (index)/(index+1) + loss.item()/(index+1)

        wandb.log({"epoch_loss": epoch_loss})


pass


if __name__ == '__main__':
    main()
