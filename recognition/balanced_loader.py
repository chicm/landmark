import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
import random
from sklearn.utils import shuffle
import settings
import settings_retrieval

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

DATA_DIR = settings.DATA_DIR

def get_classes(num_classes):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'top{}_classes.csv'.format(num_classes)))
    classes = df.classes.values.tolist()
    stoi = { classes[i]: i for i in range(len(classes))}
    return classes, stoi

def get_filename(img_id, img_dir, test_data=False, flat=False):
    if test_data:
        for i in range(10):
            fn = os.path.join(img_dir, str(i), '{}.jpg'.format(img_id))
            if os.path.exists(fn):
                return fn
        raise AssertionError('image not found: {}'.format(img_id))
    elif flat:
        return os.path.join(img_dir, '{}.jpg'.format(img_id))
    else:
        return os.path.join(img_dir, img_id[0], img_id[1], img_id[2], '{}.jpg'.format(img_id))

train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # open images mean and std
        ])

def img_augment(p=.9):
    return Compose([
        HorizontalFlip(.5),
        OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.75 ),
        Blur(blur_limit=3, p=.33),
        OpticalDistortion(p=.33),
        GridDistortion(p=.33),
        #HueSaturationValue(p=.33)
    ], p=p)

class ImageDataset(data.Dataset):
    def __init__(self, df, img_dir, stoi=None, train_mode=True, test_data=False, flat=False):
        self.input_size = 256
        self.df = df
        self.img_dir = img_dir
        self.stoi = stoi
        self.train_mode = train_mode
        self.transforms = train_transforms
        self.test_data = test_data
        self.flat = flat

    def random_select_img(self, row):
        img_ids = row['img_list']
        i = random.randint(0, len(img_ids)-1)
        return img_ids[i]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            img_id = self.random_select_img(row)
            fn = get_filename(img_id, self.img_dir, self.test_data, self.flat)
        except AssertionError:
            if self.flat:
                raise
            return torch.zeros(3, self.input_size, self.input_size), 0
        # cv2 and albumentations
        img = cv2.imread(fn)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train_mode:
            aug = img_augment(p=0.8)
            img = aug(image=img)['image']
        
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.flat:
            return img
        elif self.test_data:
            return img, 1
        else:
            return img, self.stoi[row['landmark_id']]

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        if self.flat:
            return torch.stack(batch)
        else:
            imgs = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            return imgs, labels

def get_balanced_train_val_loaders(num_classes, batch_size=4, dev_mode=False, val_num=10000, val_batch_size=1024):
    classes, stoi = get_classes(num_classes)

    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_invert.csv'))
    print(df.shape)
    df = df[df.landmark_id.isin(set(classes))]
    print(df.shape)

    df['img_list'] = df.id.map(lambda x: x.split(' '))

    df_train = df.copy()
    df_train['img_list'] = df_train.img_list.map(lambda x: x if len(x)==1 else x[1:])

    df_val = df.copy()
    df_val['img_list'] = df_val.img_list.map(lambda x: [x[0]])

    df_val_classes = shuffle(classes)[:val_num]
    df_val = df_val[df_val.landmark_id.isin(df_val_classes)]

    if dev_mode:
        df_train = df_train[:10]
        df_val = df_val[:10]

    train_set = ImageDataset(df_train, settings.TRAIN_IMG_DIR, stoi, train_mode=True)
    val_set = ImageDataset(df_val, settings.TRAIN_IMG_DIR, stoi, train_mode=False)
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = len(train_set)

    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=8, collate_fn=val_set.collate_fn, drop_last=False)
    val_loader.num = len(val_set)

    return train_loader, val_loader


def test_train_val_loader():
    train_loader, val_loader = get_balanced_train_val_loaders(50000, dev_mode=True)
    for img, label in val_loader:
        print(img.size(), img)
        print(label)
        break

if __name__ == '__main__':
    test_train_val_loader()
    #test_test_loader()
    #test_index_loader()
