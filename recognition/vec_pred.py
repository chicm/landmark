import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch.nn import DataParallel
from models import FeatureNet, create_model, FeatureNetV2
from loader import get_train_all_loader, get_test_loader
from loader_retrieval import get_retrieval_index_loader
import faiss
from tqdm import tqdm
import time

import settings
import settings_retrieval

def create_retrieval_model(args):
    model = FeatureNetV2(args.backbone, cls_model=None)
    
    if not os.path.exists(args.ckp):
        raise AssertionError('ckp not found')
    
    print('loading {}...'.format(args.ckp))
    model.load_state_dict(torch.load(args.ckp))

    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()
    model.eval()

    return model

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

def build_train_index(args):
    model = create_feature_model(args)

    train_all_loader = get_train_all_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)
    index_fn = os.path.join(settings.VECTOR_DIR, '{}.index'.format(model.name))
    build_index(args, model, train_all_loader, True, index_fn)

def build_retrieval_index(args):
    model = create_feature_model(args)

    retrieval_index_loader = get_retrieval_index_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)
    index_fn = os.path.join(settings_retrieval.VECTOR_DIR, '{}.index_retrieval'.format(model.name))
    build_index(args, model, retrieval_index_loader, False, index_fn)


def build_index(args, model, loader, loader_labeled, index_file_name, d=2048, model_output_index=None):
    #index = faiss.IndexFlatL2(2048)
    #d = 2048
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    feats = []
    with torch.no_grad():
        for batch in tqdm(loader, total=loader.num//args.batch_size):
            if loader_labeled:
                x = batch[0].cuda()
            else:
                x = batch.cuda()
            output = model(x)

            if model_output_index is None:
                feats.append(output.cpu())
            elif model_output_index == 0:
                feats.append(output[model_output_index].cpu())
            elif model_output_index == 1:
                local_feat = output[1].view(output[1].size(0), -1).cpu()
                feats.append(local_feat)

    xb = torch.cat(feats, 0).numpy()
    print('training index')
    bg = time.time()
    index.train(xb)
    train_time = time.time() - bg
    print('train time: ', train_time)
    index.add(xb)

    #index_fn = os.path.join(settings.VECTOR_DIR, '{}.index'.format(model.name))
    print('\ntotal indices: {}'.format(index.ntotal))
    print('saving index: {}'.format(index_file_name))
    faiss.write_index(index, index_file_name)

def build_retrieval_index_v2(args):
    model = create_retrieval_model(args)

    retrieval_index_loader = get_retrieval_index_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)
    index_fn = os.path.join(settings_retrieval.VECTOR_DIR, '{}.index_retrieval_2'.format(model.name))
    build_index(args, model, retrieval_index_loader, False, index_fn, d=2048, model_output_index=0)

    retrieval_index_loader = get_retrieval_index_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)
    index_fn_local = os.path.join(settings_retrieval.VECTOR_DIR, '{}.index_retrieval_2_local'.format(model.name))
    build_index(args, model, retrieval_index_loader, False, index_fn_local, d=3584, model_output_index=1)


#def pred_test_vectors()

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



def pred_retrieval(args):
    #model = create_feature_model(args)
    model = create_retrieval_model(args)

    test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode, img_size=224)

    feats = []
    founds = []
    with torch.no_grad():
        for i, (x, found) in tqdm(enumerate(test_loader), total=test_loader.num//args.batch_size):
            x = x.cuda()
            #print('x:', x.size())
            global_feat, local_feat, _ = model(x)
            #print('local:', local_feat.size())
            #exit(1)

            if args.local:
                feats.append(local_feat.view(local_feat.size(0), -1).cpu())
            else:
                feats.append(global_feat.cpu())
            founds.append(found.cpu())

    xb = torch.cat(feats, 0).numpy()
    founds = torch.cat(founds, 0).numpy()
    print(xb.shape, founds.shape)

    index_fn = args.index_fn #os.path.join(settings_retrieval.VECTOR_DIR, '{}.index_retrieval'.format(model.name))
    print('loading index...')

    index = faiss.read_index(index_fn)
    print('searching...')
    bg = time.time()
    D, I = index.search(xb, 100)
    print('search time:', time.time() - bg)

    np.save(os.path.join(settings_retrieval.VECTOR_DIR, 'D_tmp.npy'), D)
    np.save(os.path.join(settings_retrieval.VECTOR_DIR, 'I_tmp.npy'), I)
    np.save(os.path.join(settings_retrieval.VECTOR_DIR, 'founds_tmp.npy'), founds)

    #top1_index_ids = I[:, 0].squeeze()
    #print(pred_labels)
    #scores = [0.5] * xb.shape[0]
    
    create_retrieval_submission(args, I, founds, args.sub_file)

def create_retrieval_submission(args, I, founds, outfile):
    df = pd.read_csv(os.path.join(settings_retrieval.DATA_DIR, 'index_clean.csv'))
    meta = pd.read_csv(os.path.join(settings.DATA_DIR, 'test', 'test.csv'))  # use same test data as recognition

    labels = [' '.join([str(i) for i in df.iloc[x].id.values]) for x in I]

    for i in range(len(labels)):
        if founds[i] == 0:
            labels[i] = ''

    if args.dev_mode:
        meta = meta.iloc[:len(I)]  # for dev mode
        print(labels[:4])
    meta['images'] = labels
    meta.to_csv(outfile, index=False, columns=['id', 'images'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch_size')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--init_num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--build_rec', action='store_true')
    parser.add_argument('--build_ret', action='store_true')
    parser.add_argument('--task', type=str, choices=['build_rec', 'build_ret', 'pred_rec', 'pred_ret'], required=True)
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--sub_file', default='sub_vec1.csv', type=str)
    parser.add_argument('--ckp', default=None, type=str)
    parser.add_argument('--index_fn', default=None, type=str)
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)
    if args.task == 'build_rec':
        build_train_index(args)
    elif args.task == 'build_ret':
        build_retrieval_index_v2(args)
    elif args.task == 'pred_rec':
        pred_by_vector_search(args)
    elif args.task == 'pred_ret':
        pred_retrieval(args)
    else:
        pass

