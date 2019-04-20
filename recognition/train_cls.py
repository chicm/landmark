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
import settings
from loader import get_train_val_loaders, get_test_loader, get_classes
import cv2
from models import LandmarkNet, create_model
from torch.nn import DataParallel

MODEL_DIR = settings.MODEL_DIR

#tmp_one_hot = torch.eye(50000)
#tmp_one_hot.requires_grad = False

#cls_w = torch.tensor(list(range(10000))).float()  / 10000 * 10 + 1
#cls_w = cls_w.cuda()
#c = nn.CrossEntropyLoss(weight=cls_w)
c = nn.CrossEntropyLoss()

def focal_loss(x, y):
    '''Focal loss.

    Args:
    x: (tensor) sized [N,D].
    y: (tensor) sized [N,].

    Return:
    (tensor) focal loss.
        '''
    alpha = 0.25
    gamma = 2

    #t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
    #t = t[:,1:]  # exclude background
    #t = Variable(t).cuda()  # [N,20]

    t = torch.eye(50000).cuda()
    t = tmp_one_hot.index_select(0, y.cpu()).cuda()

    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    w = w.detach()
    #w.requires_grad = False
    #return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    return F.binary_cross_entropy_with_logits(x, t, w)

def criterion(args, outputs, targets):
    #return nn.CrossEntropyLoss()(outputs, targets) + focal_loss(outputs, targets) * 10
    return c(outputs, targets)
    #num_preds = torch.sigmoid(num_output)*5
    #num_loss = F.mse_loss(num_output.squeeze(), num_target.float())

    #return cls_loss + num_loss * 0.01, cls_loss.item(), num_loss.item()
    '''
    if args.focal_loss:
        return focal_loss(outputs, targets)
    else:
        return c(outputs, targets)
    '''

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


def train(args):
    print('start training...')
    model, model_file = create_model(args)
    #model = model.cuda()
    if torch.cuda.device_count() > 1:
        model_name = model.name
        model = DataParallel(model)
        model.name = model_name
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    _, val_loader = get_train_val_loaders(num_classes=args.num_classes, batch_size=args.batch_size, val_num=args.val_num)

    best_top1_acc = 0.

    print('epoch |    lr    |      %        |  loss  |  avg   |  loss  |  top1  | top10  |  best  | time |  save |')

    if not args.no_first_val:
        top10_acc, best_top1_acc, total_loss = validate(args, model, val_loader)
        print('val   |          |               |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |       |'.format(
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
        train_loader, val_loader = get_train_val_loaders(num_classes=args.num_classes, batch_size=args.batch_size, dev_mode=args.dev_mode, val_num=args.val_num)

        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target  = data
            img, target = img.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(img)
            
            loss = criterion(args, output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.6f} | {:06d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
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
            #print(img.size(), img)
            output = model(img)
            loss = criterion(args, output, target)
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

def predict_top3(args):
    model, _ = create_model(args)
    model = model.cuda()
    model.eval()
    test_loader = get_test_loader(args, batch_size=args.batch_size, dev_mode=args.dev_mode)

    preds = None
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.cuda()
            #output = torch.sigmoid(model(x))
            output, _ = model(x)
            output = F.softmax(output, dim=1)
            _, pred = output.topk(3, 1, True, True)

            if preds is None:
                preds = pred.cpu()
            else:
                preds = torch.cat([preds, pred.cpu()], 0)
            print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')

    classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)
    label_names = []
    preds = preds.numpy()
    print(preds.shape)
    for row in preds:
        label_names.append(' '.join([classes[i] for i in row]))
    if args.dev_mode:
        print(len(label_names))
        print(label_names)

    create_submission(args, label_names, args.sub_file)

def predict_softmax(args):
    model, _ = create_model(args)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.cuda()

    model.eval()
    test_loader = get_test_loader(batch_size=args.batch_size, dev_mode=args.dev_mode)

    preds = None
    scores = None
    founds = None
    with torch.no_grad():
        for i, (x, found) in enumerate(test_loader):
            x = x.cuda()
            #output = torch.sigmoid(model(x))
            #print(x[0, 0, :])
            output = model(x)
            output = F.softmax(output, dim=1)
            #pred = (output > 0.03).byte()  #  use threshold
            #preds = output.max(1, keepdim=True)[1]
            #print(output)
            #break
            score, pred = output.max(1)
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

def test_model(args):
    model, _ = create_model(args)
    model.cuda()

    
    torch.manual_seed(1234)
    x = torch.randn(2,3, 256,256).cuda()
    
    model = model.eval()
    for i in range (2):
        #model.eval()
        print(x)
        y = model(x)
        print(y)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Landmark detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=280, type=int, help='batch_size')
    parser.add_argument('--val_batch_size', default=1024, type=int, help='batch_size')
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
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--sub_file', type=str, default='sub1.csv')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    parser.add_argument('--activation', choices=['softmax', 'sigmoid'], type=str, default='softmax', help='activation')
    parser.add_argument('--val_num', default=6000, type=int, help='number of val data')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)
    #test_model(args)
    #exit(1)

    if args.predict:
        predict_softmax(args)
    else:
        train(args)
