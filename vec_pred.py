import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch.nn import DataParallel
from models import FeatureNet, create_model
from loader import get_train_all_loader
import faiss
from tqdm import tqdm
import time

import settings

def gen_train_vector(args):
    args.predict = True
    cls_model, _ = create_model(args)
    model = FeatureNet(args.backbone, cls_model=cls_model)
    
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    model.eval()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch_size')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--init_num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    gen_train_vector(args)

