import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
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

def img_augment(p=.8):
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
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=.75 ),
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

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            fn = get_filename(row['id'], self.img_dir, self.test_data, self.flat)
        except AssertionError:
            if self.flat:
                raise
            return torch.zeros(3, self.input_size, self.input_size), 0
        #print(fn)
        
        # open with PIL and transform
        #img = Image.open(fn, 'r')
        #img = img.convert('RGB')
        #img = self.transforms(img)

        # cv2 and albumentations
        img = cv2.imread(fn)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train_mode:
            aug = img_augment(p=0.8)
            img = aug(image=img)['image']
        
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #img = img.transpose((2,0,1))
        #img = (img /255).astype(np.float32)
        #print(img.shape)

        #normalize
        #mean=[0.485, 0.456, 0.406]
        #std=[0.229, 0.224, 0.225]
        
        #img[0, :,:,] = (img[0, :,:,] - mean[0]) / std[0]
        #img[1, :,:,] = (img[1, :,:,] - mean[1]) / std[1]
        #img[2, :,:,] = (img[2, :,:,] - mean[2]) / std[2]
        #img = torch.tensor(img)
        
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

def get_train_val_loaders(num_classes, batch_size=4, dev_mode=False, val_num=6000, val_batch_size=1024):
    classes, stoi = get_classes(num_classes)

    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_{}.csv'.format(num_classes)))

    split_index = int(len(df) * 0.95)
    train_df = df[:split_index]
    val_df = df[split_index:]
    if val_num is not None:
        val_df = val_df[:val_num]
    
    if dev_mode:
        train_df = train_df[:10]
        val_df = val_df[:10]
    
    #print('train df:', train_df.shape)
    #print('val df:', len(val_df))
    #print(val_df.head())
    #print(val_df.iloc[0])

    train_set = ImageDataset(train_df, settings.TRAIN_IMG_DIR, stoi, train_mode=True)
    val_set = ImageDataset(val_df, settings.TRAIN_IMG_DIR, stoi, train_mode=False)
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = len(train_set)

    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=8, collate_fn=val_set.collate_fn, drop_last=False)
    val_loader.num = len(val_set)

    return train_loader, val_loader


def get_train_all_loader(batch_size=4, dev_mode=False):
    classes, stoi = get_classes(203094)
    print('loading training data')
    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train.csv'))
    if dev_mode:
        df = df[:1000]
    print('data size:', df.shape)
    ds = ImageDataset(df, settings.TRAIN_IMG_DIR, stoi, train_mode=False)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(df)

    return loader

def get_test_loader(batch_size=1024, dev_mode=False):
    #classes, stoi = get_classes(num_classes)

    df = pd.read_csv(os.path.join(DATA_DIR, 'test', 'test.csv'))
    if dev_mode:
        df = df[:10]
    test_set = ImageDataset(df, settings.TEST_IMG_DIR, stoi=None, train_mode=False, test_data=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=test_set.collate_fn, drop_last=False)
    test_loader.num = len(test_set)

    return test_loader

def get_retrieval_index_loader(batch_size=1024, dev_mode=False):

    df = pd.read_csv(os.path.join(settings_retrieval.DATA_DIR, 'index_clean.csv'))
    if dev_mode:
        df = df[:1000]
    ds = ImageDataset(df, settings_retrieval.INDEX_IMG_DIR, stoi=None, train_mode=False, test_data=False, flat=True)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(ds)

    return loader

def test_train_val_loader():
    train_loader, val_loader = get_train_val_loaders(dev_mode=True)
    for img, label in val_loader:
        print(img.size(), img)
        print(label)
        break

def test_test_loader():
    test_loader = get_test_loader(batch_size=4, dev_mode=True)
    for img, found in test_loader:
        print(img.size(), img)
        print(found)

def test_index_loader():
    loader = get_retrieval_index_loader(batch_size=4, dev_mode=True)
    for img in loader:
        print(img.size(), img)

if __name__ == '__main__':
    #test_train_val_loader()
    #test_test_loader()
    test_index_loader()
