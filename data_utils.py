from torch.utils.data import Dataset
from pathlib import Path
import PIL

class ImageDataset(Dataset):
    def __init__(self,dir_path,transforms):
        super().__init__()
        path  = Path(dir_path)
        self.file_paths = list(path.iterdir())
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self,idx):

        path = self.file_paths[idx]
        img = PIL.Image.open(path)

        img = self.transforms(img)


        return img

    



    