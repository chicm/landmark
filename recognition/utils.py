import os
import numpy as np
import pandas as pd
import torch
import settings

def predict_model_softmax_output(model, test_loader, cls_out_index=2):
    preds = []
    scores = []
    founds = []
    cur_num = 0
    with torch.no_grad():
        for i, (x, found) in enumerate(test_loader):
            cur_num += x.size(0)
            x = x.cuda()
            output = model(x)
            cls_out = output if cls_out_index is None else output[cls_out_index]
            cls_out = F.softmax(cls_out, dim=1)
            score, pred = cls_out.max(1)

            preds.append(pred)  
            scores.append(score)
            founds.append(found)
            print('{}/{}'.format(cur_num, test_loader.num), end='\r')

    # preds is cls index, not label
    return [torch.cat(x, 0).numpy() for x in [preds, scores, founds]]

def create_submission(preds, scores, founds, outfile, classes=None, dev_mode=False):
    if classes is None:
        labels = preds
    else:
        labels = [classes[x] for x in preds]

    meta = pd.read_csv(os.path.join(settings.DATA_DIR, 'test', 'test.csv'))
    label_strs = ['{} {:.6f}'.format(i, j) for i, j in zip(labels, scores)]

    for i in range(len(label_strs)):
        if founds[i] == 0:
            label_strs[i] = ''

    if dev_mode:
        meta = meta.iloc[:len(preds)]  # for dev mode
        print(label_strs[:4])
    meta['landmarks'] = label_strs
    meta.to_csv(outfile, index=False, columns=['id', 'landmarks'])

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
