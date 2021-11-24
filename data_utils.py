from torch.utils.data import Dataset
from pathlib import Path
import PIL
import torch
import math 

class ImageDataset(Dataset):
    def __init__(self, dir_path, transforms):
        super().__init__()
        path = Path(dir_path)
        self.file_paths = list(path.iterdir())
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        path = self.file_paths[idx]
        img = PIL.Image.open(path)

        img = self.transforms(img)

        return img

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


a = torch.randn((224,224,3))
fourier_encode(a,10,4)