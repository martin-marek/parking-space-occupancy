import json
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from functools import lru_cache


class ACPDS():
    """
    A basic dataset of parking lot images,
    parking space coordinates (ROIs), and occupancies.
    Returns the tuple (image, rois, occupancy).
    """
    def __init__(self, dataset_path, ds_type='train', res=None):
        self.dataset_path = dataset_path
        self.ds_type = ds_type
        self.res = res

        # load all annotations
        with open(f'{self.dataset_path}/annotations.json', 'r') as f:
            all_annotations = json.load(f)

        # select train, valid, test, or all annotations
        if ds_type in ['train', 'valid', 'test']:
            # select train, valid, or test annotations
            annotations = all_annotations[ds_type]
        else:
            # select all annotations
            assert ds_type == 'all'
            # if using all annotations, combine the train, valid, and test dicts
            annotations = {k:[] for k in all_annotations['train'].keys()}
            for ds_type in ['train', 'valid', 'test']:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.fname_list = annotations['file_names']
        self.rois_list = annotations['rois_list']
        self.occupancy_list = annotations['occupancy_list']

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # load image
        image_path = f'{self.dataset_path}/images/{self.fname_list[idx]}'
        image = torchvision.io.read_image(image_path)
        if self.res is not None:
            image = TF.resize(image, self.res)
            
        # load occupancy
        occupancy = self.occupancy_list[idx]
        occupancy = torch.tensor(occupancy, dtype=torch.int64)

        # load rois
        rois = self.rois_list[idx]
        rois = torch.tensor(rois)
    
        return image, rois, occupancy
    
    def __len__(self):
        return len(self.fname_list)


def collate_fn(batch):
    images = [item[0] for item in batch]
    rois = [item[1] for item in batch]
    occupancy = [item[2] for item in batch]
    return [images, rois, occupancy]


def create_datasets(dataset_path, *args, **kwargs):
    """
    Create training and test DataLoaders.
    Returns the tuple (image, rois, occupancy).
    During the first pass, the DataLoaders will be cached.
    """
    ds_train = ACPDS(dataset_path, 'train', *args, **kwargs)
    ds_valid = ACPDS(dataset_path, 'valid', *args, **kwargs)
    ds_test = ACPDS(dataset_path, 'test', *args, **kwargs)
    data_loader_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(ds_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return data_loader_train, data_loader_valid, data_loader_test