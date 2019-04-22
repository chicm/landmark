#import sys
import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
import random
#sys.path.append('../recognition')
import settings_retrieval

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

DATA_DIR = settings_retrieval.DATA_DIR

exclude_ids = set([1307, 2204, 2216, 2952, 3022, 4753, 6131, 6798, 8615, 9208, 9519, 10151, 14134, 14951])

def get_filename(img_id, img_dir):
    return os.path.join(img_dir, '{}.jpg'.format(img_id))

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
    def __init__(self, df, invert_dict, img_dir, train_mode=True, test_data=False):
        self.input_size = 256
        self.df = df
        self.invert_dict = invert_dict
        self.img_dir = img_dir
        self.train_mode = train_mode
        self.test_data = test_data
        self.num_labels = 14952 #len(self.invert_dict)

    def get_img(self, img_id):
        fn = get_filename(img_id, self.img_dir)
        # cv2 and albumentations
        img = cv2.imread(fn)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train_mode:
            aug = img_augment(p=0.8)
            img = aug(image=img)['image']
        
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def get_random_label(self, excludes=[]):
        label = random.randint(0, self.num_labels-1)
        while label in excludes or label in exclude_ids:
            label = random.randint(0, self.num_labels-1)
        return label

    def __getitem__(self, index):
        row = self.df.iloc[index]
        anchor_label = row['landmark_id']
        anchor_img_id = row['id']

        if self.test_data:
            return self.get_img(anchor_img_id)

        pos_img_id = random.choice(self.invert_dict[anchor_label])

        neg1_label = self.get_random_label(excludes=[anchor_label])
        #(anchor_label + random.randint(1, self.num_labels // 2)) % self.num_labels
        neg2_label = self.get_random_label(excludes=[anchor_label, neg1_label])
        #(anchor_label + random.randint(self.num_labels // 2, self.num_labels-2)) % self.num_labels
        neg1_img_id = random.choice(self.invert_dict[neg1_label])
        neg2_img_id = random.choice(self.invert_dict[neg2_label])

        imgs = [self.get_img(x) for x in [anchor_img_id, pos_img_id, neg1_img_id, neg2_img_id]]
        
        return imgs, [anchor_label, anchor_label, neg1_label, neg2_label]

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        if self.test_data:
            return torch.stack(batch)
        else:
            #imgs = torch.stack([x[0] for x in batch])
            #labels = torch.tensor([x[1] for x in batch])
            #return imgs, labels

            batch_size = len(batch)
            images = []
            labels = []
            for b in range(batch_size):
                if batch[b][0] is None:
                    continue
                else:
                    images.extend(batch[b][0])
                    labels.extend(batch[b][1])
            images = torch.stack(images, 0)
            labels = torch.from_numpy(np.array(labels))
            return images, labels


def get_train_val_loaders(batch_size=4, dev_mode=False, val_num=1000, val_batch_size=1024):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train_clean.csv'))
    df_invert = pd.read_csv(os.path.join(DATA_DIR, 'train_clean_invert.csv'))
    df_invert['img_list'] = df_invert.id.map(lambda x: x.split(' '))
    invert_dict = pd.Series(df_invert.img_list.values, index=df_invert.landmark_id).to_dict()
    #print(len(invert_dict))
    #print(invert_dict[14134])
    #exclude_classes = []
    #for i in range(14952):
    #    if i not in invert_dict:
    #        exclude_classes.append(i)
    #print(exclude_classes)
    #exit(0)

    split_index = int(len(df) * 0.95)
    train_df = df[:split_index]
    val_df = df[split_index:]
    if val_num is not None:
        val_df = val_df[:val_num]

    if dev_mode:
        train_df = train_df[:10]
        val_df = val_df[:10]

    train_set = ImageDataset(train_df, invert_dict, settings_retrieval.TRAIN_IMG_DIR, train_mode=True)
    val_set = ImageDataset(val_df, invert_dict, settings_retrieval.TRAIN_IMG_DIR, train_mode=False)
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = len(train_set)

    val_loader = data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn, drop_last=False)
    val_loader.num = len(val_set)

    return train_loader, val_loader


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
    train_loader, val_loader = get_train_val_loaders(dev_mode=True, batch_size=2)
    for img, label in train_loader:
        print(img.size())
        print(label.size())
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
    test_train_val_loader()
    #test_test_loader()
    #test_index_loader()
