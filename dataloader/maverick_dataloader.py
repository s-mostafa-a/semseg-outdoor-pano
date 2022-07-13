import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class Maverick(Dataset):
    """Maverick Panoramic Dataloader"""

    def __init__(self, args):
        super().__init__()
        self._base_dir = args.dataset_path
        self._image_dir = os.path.join(self._base_dir, 'test', 'rgb')
        self.args = args
        self.images = []
        self.images = glob.glob(self._image_dir + "/*.png")
        print('Number of images in {}: {:d}'.format('test set', len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.transform_image(img)
        sample = {'image': img, 'file_name': self.images[index].split("/")[-1].split(".")[0]}
        return sample

    def transform_image(self, image):
        img = image.resize((self.args.img_witdh, self.args.img_height), Image.BILINEAR)
        img = self.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

    def normalize(self, image, mean=None, std=None):
        img = image
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img
