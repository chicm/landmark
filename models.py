import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from net.senet import se_resnext50_32x4d, se_resnet50, senet154, se_resnet152, se_resnext101_32x4d, se_resnet101
from net.densenet import densenet121, densenet161, densenet169, densenet201
import settings


class LandmarkNet(nn.Module):
    def __init__(self, backbone_name, num_classes=1000, pretrained=True):
        super(LandmarkNet, self).__init__()
        print('num_classes:', num_classes)
        if backbone_name in ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet101', 'se_resnet50', 'senet154', 'se_resnet152']:
            self.backbone = eval(backbone_name)()
        elif backbone_name in ['resnet34', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']:
            self.backbone = eval(backbone_name)(pretrained=pretrained)
        else:
            raise ValueError('unsupported backbone name {}'.format(backbone_name))

        if backbone_name == 'resnet34':
            ftr_num = 512
        elif backbone_name == 'densenet161':
            ftr_num = 2208
        elif backbone_name == 'densenet121':
            ftr_num = 1024
        elif backbone_name == 'densenet169':
            ftr_num = 1664
        elif backbone_name == 'densenet201':
            ftr_num = 1920
        else:
            ftr_num = 2048
        self.ftr_num = ftr_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(ftr_num, num_classes)
        self.name = 'LandmarkNet_{}_{}'.format(backbone_name, num_classes)
    
    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, 0.4, self.training)
        x = x.view(x.size(0), -1)
        return self.logit(x)
    
    def forward(self, x):
        x = self.backbone.features(x)
        return self.logits(x)


def create_model(args):
    if args.init_ckp is not None:
        model = LandmarkNet(backbone_name=args.backbone, num_classes=args.init_num_classes)
        model.load_state_dict(torch.load(args.init_ckp))
        if args.init_num_classes != args.num_classes:
            model.logit = nn.Linear(model.ftr_num, args.num_classes)
            model.name = 'LandmarkNet_{}_{}'.format(args.backbone, args.num_classes)
    else:
        model = LandmarkNet(backbone_name=args.backbone, num_classes=args.num_classes)

    model_file = os.path.join(settings.MODEL_DIR, model.name, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    
    return model, model_file

from argparse import Namespace

def test():
    x = torch.randn(2, 3, 256, 256).cuda()
    args = Namespace()
    args.init_ckp = None
    args.backbone = 'se_resnet50'
    args.ckp_name = 'best_model.pth'
    args.predict = False

    model = create_model(args)[0].cuda()
    y = model(x)
    print(y.size(), y)

if __name__ == '__main__':
    test()
    #convert_model4()
