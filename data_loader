import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image


class MarineSnowDataset(Dataset):
    def __init__(self, background_dir, marine_snow_dir, transform=None):
        self.background_dir = background_dir
        self.marine_snow_dir = marine_snow_dir
        self.transform = transform

        self.background_images = sorted(os.listdir(background_dir))
        self.marine_snow_images = sorted(os.listdir(marine_snow_dir))

    def __len__(self):
        return len(self.marine_snow_images)
    
    def extract_background_name(self, snow_name):

        # eg snow_image_name = bg0_aug_0_snow_1829.png
        items = snow_name.split('_')[:3]
        bg_name = '_'.join(items) + '.png' 
        return bg_name
    
    def find_corresponding_background(self, snow_name):
        bg_name = self.extract_background_name(snow_name)
        for bg_image in self.background_images:
            if bg_name in bg_image:
                return os.path.join(self.background_dir, bg_image)
        return None
        
    def __getitem__(self, index):
        marine_snow_name = self.marine_snow_images[index]

        marine_snow_path = os.path.join(self.marine_snow_dir, marine_snow_name)

        background_path = self.find_corresponding_background(marine_snow_name)
        if background_path is None:
            raise FileNotFoundError(f"Background image for {marine_snow_name} not found.")
        
        background_image = Image.open(background_path).convert('RGB')
        marine_snow_image = Image.open(marine_snow_path).convert('RGB')

        if self.transform:
            background_image = self.transform(background_image)
            marine_snow_image = self.transform(marine_snow_image)

        return marine_snow_image, background_image, marine_snow_name
