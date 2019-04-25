import os
import argparse
import numpy as np
import pandas as pd
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import settings
from loader import get_train_val_loaders, get_test_loader, get_classes
import cv2
from models import LandmarkNet, create_model
from torch.nn import DataParallel

MODEL_DIR = settings.MODEL_DIR

def create_models_from_ckps(args):
    models = []
    print(args.ckps)
    ckps = args.ckps.split(',')
    for ckp_fn in ckps:
        model_args = args
        model_args.ckp_name = os.path.basename(ckp_fn)
        model_args.predict = True
        model_name = os.path.basename(os.path.dirname(ckp_fn))
        model_args.suffix_name = model_name.split('_')[0]
        model_args.num_classes = int(model_name.split('_')[-1])
        model_args.start_index = int(model_name.split('_')[-2])
        model_args.backbone = '_'.join(model_name.split('_')[1:-2])
        print(model_args.suffix_name, model_args.backbone, model_args.start_index, model_args.num_classes)
        model, _ = create_model(model_args)

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model = model.cuda()
        model.eval()

        models.append(model)
    return models

def ensemble_predict(args):
    models = create_models_from_ckps(args)

    test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)

    preds = None
    scores = None
    founds = None
    with torch.no_grad():
        for i, (x, found) in enumerate(test_loader):
            x = x.cuda()
            #output = torch.sigmoid(model(x))
            outputs = []
            for model in models:
                output = model(x)
                output = F.softmax(output, dim=1)
                outputs.append(output)
            avg_ouput = torch.stack(outputs).mean(0)
            #print(x[0, 0, :])
            #output = model(x)
            #output = F.softmax(output, dim=1)
            #pred = (output > 0.03).byte()  #  use threshold
            #preds = output.max(1, keepdim=True)[1]
            #print(output)
            #break
            score, pred = avg_ouput.max(1)
            #print(pred.size())

            if preds is None:
                preds = pred.cpu()
            else:
                preds = torch.cat([preds, pred.cpu()], 0)
            
            if scores is None:
                scores = score.cpu()
            else:
                scores = torch.cat([scores, score.cpu()], 0)

            if founds is None:
                founds = found
            else:
                founds = torch.cat([founds, found], 0)

            print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')

    classes, stoi = get_classes(num_classes=args.num_classes)
    preds = preds.numpy()
    scores = scores.numpy()
    print(preds.shape)

    pred_labels = [classes[i] for i in preds]
    
    create_submission(args, pred_labels, scores, founds, args.sub_file)

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
    parser.add_argument('--batch_size', default=1024, type=int, help='batch_size')
    parser.add_argument('--num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckps', type=str, required=True)
    parser.add_argument('--init_ckp', type=str, default=None)
    parser.add_argument('--sub_file', type=str, default='ensemble1.csv')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)
    #create_models_from_ckps(args)
    ensemble_predict(args)
