# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import torchvision.transforms as transforms
from backbone.vit import vit_base_16

import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch.optim





class Domainnet(Dataset):

    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        train_txt = './datasets/train.txt'
        test_txt = './datasets/test.txt'

        if self.train:
            train_images = []
            train_labels = []
            with open(train_txt, 'r') as dict_file:
                for line in dict_file:
                    (value, key) = line.strip().split(' ')
                    # train_images.append(np.array(Image.open(os.path.join(root, value)).convert('RGB').resize((256,256))))
                    train_labels.append(int(key))
            # train_images = np.array(train_images)
            train_images = np.load("./train_imgs.npy", allow_pickle=True)
            train_labels = np.array(train_labels)
            # import pdb;pdb.set_trace()
            self.data = train_images
            self.targets = train_labels
        else:
            test_images = []
            test_labels = []
            with open(test_txt, 'r') as dict_file:
                for line in dict_file:
                    (value, key) = line.strip().split(' ')
                    # test_images.append(np.array(Image.open(os.path.join(root, value)).convert('RGB').resize((256,256))))
                    test_labels.append(int(key))
            # test_images = np.array(test_images)
            # import pdb;pdb.set_trace()

            test_images = np.load("./test_imgs.npy", allow_pickle=True)
            test_labels = np.array(test_labels)
            self.data = test_images
            self.targets = test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img), mode='RGB')
        # img = Image.open(img).convert('RGB')

        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyDomainnet(Domainnet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyDomainnet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img), mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img


class SequentialDomainnet(ContinualDataset):

    NAME = 'seq-domainnet'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             self.get_normalization_transform()])

        train_dataset = MyDomainnet(base_path() + 'domainnet',
                                 train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = Domainnet(base_path() + 'domainnet',
                        train=False, download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_backbone():
        return vit_base_16(SequentialDomainnet.N_CLASSES_PER_TASK
                        * SequentialDomainnet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None