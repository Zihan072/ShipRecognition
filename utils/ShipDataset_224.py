import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ShipDataset(Dataset):

    def __init__(self, path_pos, path_neg, n_samples=None, is_train=True, target_dim=1):
        assert (is_train or len(path_pos) == len(path_neg))
        assert (target_dim in [1, 2])
        self.is_train = is_train
        self.target_dim = target_dim
        self.n_samples = n_samples
        self.path_pos = path_pos
        self.path_neg = path_neg

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(180, translate=(0, 0), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ])
        self.process = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),  #into 224*224
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def augment(self, img):
        if self.is_train:
            return self.process(self.augmentation(img))
        else:
            return self.process(img)

    def get_target(self, is_ship):
        if self.target_dim == 1:
            return torch.tensor([int(is_ship)], dtype=torch.float32)
        if self.target_dim == 2:
            x = [1, 0] if is_ship else [0, 1]
            return torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        if self.is_train:
            return self.n_samples
        else:
            return len(self.path_pos) + len(self.path_neg)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            i = (idx // 2) % len(self.path_pos)
            return self.augment(load_image(self.path_pos[i])), self.get_target(True)
        else:
            i = (idx // 2) % len(self.path_neg)
            return self.augment(load_image(self.path_neg[i])), self.get_target(False)


def get_ship_dataset(path, n_train, validation=0., target_dim=1):
    path_pos, path_neg = [], []
    for file in os.listdir(path):
        full = os.path.join(path, file)
        if file[0] == '0':
            path_neg.append(full)
        if file[0] == '1':
            path_pos.append(full)

    if validation > 0:
        n_val = int((len(path_pos) + len(path_neg)) * validation * 0.5)
        ds_train = ShipDataset(path_pos[:-n_val], path_neg[:-n_val], n_train, target_dim=target_dim)
        ds_val   = ShipDataset(path_pos[-n_val:], path_neg[-n_val:], is_train=False, target_dim=target_dim)
        return ds_train, ds_val
    else:
        ds_train = ShipDataset(path_pos, path_neg, n_train)
        return ds_train

