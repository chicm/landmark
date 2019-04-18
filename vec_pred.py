import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch.nn import DataParallel
from models import FeatureNet, create_model
from loader import get_train_all_loader, get_test_loader
import faiss
from tqdm import tqdm
import time

import settings

def create_feature_model(args):
    args.predict = True
    cls_model, _ = create_model(args)
    model = FeatureNet(args.backbone, cls_model=cls_model)
    
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    model.eval()
    return model

def gen_train_vector(args):
    model = create_feature_model(args)

    train_all_loader = get_train_all_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)

    #index = faiss.IndexFlatL2(2048)
    d = 2048
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    outputs = []
    with torch.no_grad():
        for i, (x, label) in tqdm(enumerate(train_all_loader), total=train_all_loader.num//args.batch_size):
            x = x.cuda()
            output = model(x)
            #index.add(output.cpu().numpy())
            #if outputs is None:
            #    outputs = output.cpu()
            #else:
            #    outputs = torch.cat([outputs, output.cpu()], 0)
            #print('{}/{}'.format(args.batch_size*(i+1), train_all_loader.num), end='\r')
            outputs.append(output.cpu())
    xb = torch.cat(outputs, 0).numpy()
    print('training index')
    bg = time.time()
    index.train(xb)
    train_time = time.time() - bg
    print('train time: ', train_time)
    index.add(xb)

    index_fn = os.path.join(settings.VECTOR_DIR, '{}.index'.format(model.name))
    print('\ntotal indices: {}'.format(index.ntotal))
    print('saving index: {}'.format(index_fn))
    faiss.write_index(index, index_fn)

def pred_by_vector_search(args):
    model = create_feature_model(args)

    test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)

    outputs = []
    founds = []
    with torch.no_grad():
        for i, (x, found) in tqdm(enumerate(test_loader), total=test_loader.num//args.batch_size):
            x = x.cuda()
            output = model(x)

            outputs.append(output.cpu())
            founds.append(found.cpu())

    xb = torch.cat(outputs, 0).numpy()
    founds = torch.cat(founds, 0).numpy()
    print(xb.shape, founds.shape)

    index_fn = os.path.join(settings.VECTOR_DIR, '{}.index'.format(model.name))
    print('loading index...')
    index = faiss.read_index(index_fn)
    print('searching...')
    D, I = index.search(xb, 10)

    top1_index_ids = I[:, 0].squeeze()
    #print('I:', I)
    #print('top1 index ids:', top1_index_ids)

    pred_labels = get_labels(top1_index_ids)
    #print(pred_labels)
    scores = [0.5] * xb.shape[0]
    
    create_submission(args, pred_labels, scores, founds, args.sub_file)

def get_labels(index_ids):
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'train', 'train.csv'))
    df_res = df.iloc[index_ids]

    return df_res.landmark_id.values

def create_submission(args, predictions, scores, founds, outfile):
    meta = pd.read_csv(os.path.join(settings.DATA_DIR, 'test', 'test.csv'))
    labels = ['{} {:.6f}'.format(i, j) for i, j in zip(predictions, scores)]

    for i in range(len(labels)):
        if founds[i] == 0:
            labels[i] = ''

    if args.dev_mode:
        meta = meta.iloc[:len(predictions)]  # for dev mode
        print(labels[:4])
    meta['landmarks'] = labels
    meta.to_csv(outfile, index=False, columns=['id', 'landmarks'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch_size')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--init_num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--sub_file', default='sub_vec1.csv', type=str)
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)
    if args.gen:
        gen_train_vector(args)
    else:
        pred_by_vector_search(args)
