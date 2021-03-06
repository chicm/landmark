import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.utils import shuffle
import settings
import settings_retrieval

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Resize, RandomSizedCrop,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

class Rotate90(RandomRotate90):
    def apply(self, img, factor=1, **params):
        return np.ascontiguousarray(np.rot90(img, factor))

DATA_DIR = settings.DATA_DIR

def get_classes(num_classes, start_index=0, other=False):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'top203094_classes.csv'))
    classes = df.classes.values.tolist()[start_index: start_index+num_classes]
    if other:
        classes.append(-1)
    assert num_classes == len(classes)
    stoi = { classes[i]: i for i in range(len(classes))}
    return classes, stoi

def get_filename(img_id, img_dir, test_data=False, flat=False, stage2=False):
    if stage2:
        return os.path.join(img_dir, img_id[0], img_id[1], img_id[2], '{}.jpg'.format(img_id))
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

def weak_augment(p=.8):
    return Compose([
        RandomSizedCrop((200, 250), 256, 256, p=0.8),
        RandomRotate90(p=0.05),
        OneOf([
                #CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=.75 ),
        Blur(blur_limit=3, p=.33),
        OpticalDistortion(p=.33),
        #GridDistortion(p=.33),
        #HueSaturationValue(p=.33)
    ], p=p)

def get_tta_aug_old(tta_index=0):
    if tta_index == 0:
        return Compose([Resize(256, 256)], p=1.)
    else:
        return Compose([RandomSizedCrop((200, 250), 256, 256, p=1.)], p=1.0)

def get_tta_aug(tta_index=None):
    tta_augs = {
        1: [HorizontalFlip(always_apply=True)],
        2: [VerticalFlip(always_apply=True)],
        3: [HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True)],
        4: [Rotate90(always_apply=True)],
        5: [Rotate90(always_apply=True), HorizontalFlip(always_apply=True)],
        6: [VerticalFlip(always_apply=True), Rotate90(always_apply=True)],
        7: [HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True), Rotate90(always_apply=True)],
    }
    
    return Compose(tta_augs[tta_index], p=1.0)


class ImageDataset(data.Dataset):
    def __init__(self, df, img_dir, train_mode=True, test_data=False, flat=False, input_size=256, tta_index=None, stage2=False):
        self.input_size = input_size
        self.df = df
        self.img_dir = img_dir
        self.train_mode = train_mode
        self.transforms = train_transforms
        self.test_data = test_data
        self.flat = flat
        self.tta_index = tta_index
        self.stage2 = stage2

    def get_img(self, fn):
        # open with PIL and transform
        #img = Image.open(fn, 'r')
        #img = img.convert('RGB')
        #img = self.transforms(img)

        # cv2 and albumentations
        img = cv2.imread(fn)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.stage2:
            img = cv2.resize(img, (256, 256))
        elif self.train_mode:
            #aug = img_augment(p=0.8)
            aug = weak_augment(p=0.8)
            img = aug(image=img)['image']
        elif self.tta_index is not None and self.tta_index > 0:
            aug = get_tta_aug(self.tta_index)
            img = aug(image=img)['image']

        #if self.input_size != 256:
        #    aug = resize_aug(p=1.)
        #    img = aug(image=img)['image']
        
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
        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            fn = get_filename(row['id'], self.img_dir, self.test_data, self.flat, self.stage2)
        except AssertionError:
            if self.flat or self.stage2:
                raise
            return torch.zeros(3, self.input_size, self.input_size), 0
        #print(fn)
        
        img = self.get_img(fn)
        
        if self.flat:
            return img
        elif self.test_data:
            return img, 1
        else:
            return img, row['label']

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        if self.flat:
            return torch.stack(batch)
        else:
            imgs = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            return imgs, labels

def get_train_val_loaders(num_classes, start_index=0, batch_size=4, dev_mode=False, val_num=6000, val_batch_size=1024, other=False, tta_index=None):
    classes, stoi = get_classes(num_classes, start_index=start_index, other=other)

    train_df = None
    val_df = None

    if num_classes == 50000 and start_index == 0:
        df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_{}.csv'.format(num_classes)))
        df['label'] = df.landmark_id.map(lambda x: stoi[x])
    elif num_classes == 203094:
        df_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train.csv'))
        df = shuffle(df_all, random_state=1234)
        df['label'] = df.landmark_id.map(lambda x: stoi[x])
    else:
        df_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train.csv'))
        df_selected = shuffle(df_all[df_all.landmark_id.isin(set(classes))].copy().sort_values(by='id'), random_state=1234)
        df_selected['label'] = df_selected.landmark_id.map(lambda x: stoi[x])

        split_index = int(len(df_selected) * 0.95)
        train_df = df_selected[:split_index]
        val_df = df_selected[split_index:]
        if val_num is not None:
            val_df = val_df[:val_num]

        if other:
            df_other = df_all[~df_all.landmark_id.isin(set(classes))].sample(1000)
            df_other['label'] = num_classes-1 # TODO handle this at prediction

            train_df = pd.concat([train_df, df_other], sort=False)

        #print(df.shape, df_selected.shape, df_other.shape)
        #print(df.head(20))
    

    if train_df is None:
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

    train_set = ImageDataset(train_df, settings.TRAIN_IMG_DIR, train_mode=True)
    val_set = ImageDataset(val_df, settings.TRAIN_IMG_DIR, train_mode=False, tta_index=tta_index)
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = len(train_set)

    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=8, collate_fn=val_set.collate_fn, drop_last=False)
    val_loader.num = len(val_set)
    val_loader.labels = val_df.label.values

    return train_loader, val_loader


def get_train_all_loader(batch_size=4, dev_mode=False):
    classes, stoi = get_classes(203094)
    print('loading training data')
    df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train.csv'))
    df['label'] = df.landmark_id.map(lambda x: stoi[x])
    if dev_mode:
        df = df[:1000]
    print('data size:', df.shape)
    ds = ImageDataset(df, settings.TRAIN_IMG_DIR, train_mode=False)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(df)

    return loader

def get_test_loader(batch_size=1024, dev_mode=False, img_size=256):
    #classes, stoi = get_classes(num_classes)

    df = pd.read_csv(os.path.join(DATA_DIR, 'test', 'test.csv'))
    if dev_mode:
        df = df[:10]
    test_set = ImageDataset(df, settings.TEST_IMG_DIR, train_mode=False, test_data=True, input_size=img_size)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=test_set.collate_fn, drop_last=False)
    test_loader.num = len(test_set)

    return test_loader

def get_stage2_test_loader(batch_size=1024, dev_mode=False):
    df = pd.read_csv(os.path.join(settings_retrieval.DATA_DIR, 'stage2', 'test.csv'))
    img_dir = os.path.join(settings_retrieval.DATA_DIR, 'stage2', 'test')
    if dev_mode:
        df = df[:100]
    ds = ImageDataset(df, img_dir, train_mode=False, test_data=True, stage2=True)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(ds)

    return loader



def test_train_val_loader():
    train_loader, val_loader = get_train_val_loaders(50000, 50000, dev_mode=True)
    print(val_loader.labels.shape)
    print(val_loader.labels[:5])
    for img, label in val_loader:
        print(img.size(), img)
        print(label)
        break

def test_test_loader():
    test_loader = get_test_loader(batch_size=4, dev_mode=True, img_size=224)
    for img, found in test_loader:
        print(img.size(), img)
        print(found)

def test_index_loader():
    loader = get_retrieval_index_loader(batch_size=4, dev_mode=True)
    for img in loader:
        print(img.size(), img)

def test_stage2_test_loader():
    loader = get_stage2_test_loader(batch_size=4, dev_mode=True)
    for img, _ in loader:
        print(img.size(), img)


if __name__ == '__main__':
    #test_train_val_loader()
    #test_test_loader()
    #test_index_loader()
    test_stage2_test_loader()
