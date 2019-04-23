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
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau

from loader_retrieval import get_train_val_loaders
from balanced_loader import get_balanced_train_val_loaders
import cv2
from models import FeatureNetV2, create_model
from torch.nn import DataParallel
from triplet_loss import global_loss, local_loss, TripletLoss
import settings_retrieval

MODEL_DIR = settings_retrieval.MODEL_DIR

c = nn.CrossEntropyLoss()


def criterion(args, outputs, targets):
    #return nn.CrossEntropyLoss()(outputs, targets) + focal_loss(outputs, targets) * 10
    return c(outputs, targets)

def accuracy(output, label, topk=(1,10)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res

def create_retrieval_model(args):
    cls_model = None
    if args.from_cls_model:
        cls_model, cls_model_file = create_model(args)
    model = FeatureNetV2(args.backbone, cls_model=cls_model)

    model_file = os.path.join(MODEL_DIR, model.name, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))

    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    return model, model_file

def get_loss(global_feat, local_feat, results, labels):
    triple_loss = global_loss(TripletLoss(margin=0.6), global_feat, labels)[0] + \
                  local_loss(TripletLoss(margin=0.6), local_feat, labels)[0]
    celoss = c(results, labels)
    #print('train result:', results.size())
    #print('loss:', celoss.mean())

    return triple_loss + celoss, triple_loss.item(), celoss.item()

def train(args):
    print('start training...')
    model, model_file = create_retrieval_model(args)

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    _, val_loader = get_train_val_loaders(batch_size=args.batch_size, val_batch_size=args.val_batch_size, val_num=args.val_num)

    best_top1_acc = 0.

    print('epoch |    lr    |      %        |  closs  |  tloss  |  avg   |  loss  |  top1  | top10  |  best  | time |  save |')

    if not args.no_first_val:
        top10_acc, best_top1_acc, total_loss = validate(args, model, val_loader)
        print('val   |          |               |         |         |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |       |'.format(
            total_loss, best_top1_acc, top10_acc, best_top1_acc))

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_top1_acc)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loader, val_loader = get_train_val_loaders(batch_size=args.batch_size, dev_mode=args.dev_mode, val_batch_size=args.val_batch_size, val_num=args.val_num)
        train_loss = 0
        total_trip_loss = 0.

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target  = data
            img, target = img.cuda(), target.cuda().long()
            optimizer.zero_grad()
            global_feat, local_feat, results = model(img)

            batch_loss, trip_loss, ce_loss = get_loss(global_feat, local_feat, results, target)
            #batch_loss = model.loss

            #loss = criterion(args, output, target)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()

            #pred = F.softmax(results, dim=1)
            #top1, top10 = accuracy(pred, target)
            #top1, top10 = top1/len(img), top10/len(img)
            #print('top1:', top1, 'top10:', top10)

            train_loss += batch_loss.item()
            total_trip_loss += trip_loss
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1),
                train_loader.num, ce_loss, total_trip_loss/(batch_idx+1), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), model_file+'_latest')
                else:
                    torch.save(model.state_dict(), model_file+'_latest')

                top10_acc, top1_acc, total_loss = validate(args, model, val_loader)
                
                _save_ckp = ''
                if args.always_save or top1_acc > best_top1_acc:
                    best_top1_acc = top1_acc
                    if isinstance(model, DataParallel):
                        torch.save(model.module.state_dict(), model_file)
                    else:
                        torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
                    total_loss, top1_acc, top10_acc, best_top1_acc, (time.time() - bg) / 60, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(top1_acc)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    #del model, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader):
    model.eval()
    #print('validating...')

    #total_num = 0
    top1_corrects, corrects = 0, 0
    total_loss = 0.
    n_batches = 0
    with torch.no_grad():
        for img, target in val_loader:
            n_batches += 1
            img, target = img.cuda(), target.cuda()
            #print('img:', img.size(), target.size())
            global_feat, local_feat, output = model(img)
            #print('output:', output.size())
            loss = criterion(args, output, target)
            #print('batch loss:', loss.mean())
            #exit(0)
            total_loss += loss.item()

            #print(output.size(), output)
            #break
            
            #preds = output.max(1, keepdim=True)[1]
            #corrects += preds.eq(target.view_as(preds)).sum().item()
            output = F.softmax(output, dim=1)
            top1, top10 = accuracy(output, target)
            top1_corrects += top1
            corrects += top10
            #total_num += len(img)
    #print('top 10 corrects:', corrects)
    #print('top 1 corrects:', top1_corrects)
            
    top10_acc = corrects / val_loader.num
    top1_acc = top1_corrects / val_loader.num
    #n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    return top10_acc, top1_acc, total_loss / n_batches
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--from_cls_model', action='store_true')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=20, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=200, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--init_num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--num_classes', type=int, default=50000, help='init num classes')
    parser.add_argument('--start_index', type=int, default=0, help='class start index')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--sub_file', type=str, default='sub1.csv')
    parser.add_argument('--suffix_name', type=str, default='LandmarkNet')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--val_num', default=6000, type=int, help='number of val data')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)

    train(args)
