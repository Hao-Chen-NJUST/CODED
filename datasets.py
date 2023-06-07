# -*- coding: utf-8 -*-
import glob
import random
import torch
import math
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from util import collate_fn

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class ImageDataset(Dataset):
    def __init__(self, args, root, mode='train', unlabelled=False):
        self.img_size = args.img_size
        self.crop_size = args.crop_size

        assert self.img_size >= self.crop_size
        self.transform_train = transforms.Compose([transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
                                                   transforms.Pad(int(self.img_size / 10), fill=0,
                                                                  padding_mode='constant'),
                                                   transforms.RandomRotation(10),
                                                   transforms.RandomCrop((self.crop_size, self.crop_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                   ])

        self.args = args
        self.mode = mode
        if mode == 'train' and not unlabelled:
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/good/*.*'))
        elif mode == 'train' and unlabelled:
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/unlabelled/*.*'))
        elif mode == 'test':
            self.files = sorted(glob.glob(os.path.join(root, mode) + '/*/*.*'))

    def _align_transform(self, img):
        # resize to 224
        img = TF.resize(img, self.crop_size, Image.BICUBIC)

        # toTensor
        img = TF.to_tensor(img)

        # normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img

    def __getitem__(self, index):
        filename = self.files[index]
        img = Image.open(filename)
        img = img.convert('RGB')
        W, H = img.size

        if self.mode == 'train':
            img = self.transform_train(img)
            if 'good' in filename:
                return img, 0
            elif 'unlabelled' in filename:
                return img, 0.5

        elif self.mode == 'test':
            transform_test = self._align_transform
            img = transform_test(img)
            _, h, w = img.size()

            if 'good' in filename:
                # annos = []
                return img, 0, filename
            else:
                # with open(filename.replace('test', 'annotations').replace('.jpg', '.json'), 'r') as f:
                #     annos = json.load(f)
                # bbox = []
                # for i in range(annos['shapes'].__len__()):
                #     temp = annos['shapes'][i]['points']
                #     temp[0][0] = temp[0][0] * w / W
                #     temp[1][0] = temp[1][0] * w / W
                #     temp[0][1] = temp[0][1] * h / H
                #     temp[1][1] = temp[1][1] * h / H
                #     bbox.append(temp)
                return img, 1, filename

    def __len__(self):
        return len(self.files)


# Configure dataloaders
def Get_dataloader(args):
    train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name), mode='train'),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=True, collate_fn=collate_fn)

    test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name), mode='test'),
                                 batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
                                 drop_last=False, collate_fn=collate_fn)
    train_unlabelled_dataloader = None
    if args.unlabelled:
        train_unlabelled_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root, args.dataset_name),
                                                              mode='train', unlabelled=True),
                                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                 drop_last=True, collate_fn=collate_fn)

    return train_dataloader, train_unlabelled_dataloader, test_dataloader
