import os
from torch.utils.data import Dataset, random_split
import pandas as pd
from PIL import Image

class CustomDataSet(Dataset):
    def __init__(self, metadata, dir_root,transform= None):
        self.dir_root= dir_root
        self.annotations= pd.read_csv(metadata, header= 1)
        self.transform= transform
        
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        fileName= self.annotations.iloc[idx, 0]
        im= os.path.join(self.dir_root, fileName)
        img = Image.open(im).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        classCategory= self.annotations.iloc[idx, 1]
        # group= self.annotations.iloc[idx, 2]
        # im_group= self.group_array[idx]
        # split= self.annotations.iloc[idx, 3] # 0: train, 1: valid, 2: test
        return img, im, classCategory