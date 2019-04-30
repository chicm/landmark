import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from net.senet import se_resnext50_32x4d, se_resnet50, senet154, se_resnet152, se_resnext101_32x4d, se_resnet101
from net.densenet import densenet121, densenet161, densenet169, densenet201
from net.nasnet import nasnetalarge
from net.inceptionresnetv2 import inceptionresnetv2
from net.inceptionv4 import inceptionv4
from net.dpn import dpn98, dpn107, dpn92, dpn131

from triplet_loss import global_loss, local_loss, TripletLoss
import functional as LF

import settings

c = nn.CrossEntropyLoss()

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def get_num_features(backbone_name):
    if backbone_name in ['resnet18', 'resnet34']:
        ftr_num = 512
    elif backbone_name =='nasnetmobile':
        ftr_num = 1056
    elif backbone_name == 'mobilenet':
        ftr_num = 1280
    elif backbone_name == 'densenet161':
        ftr_num = 2208
    elif backbone_name == 'densenet121':
        ftr_num = 1024
    elif backbone_name == 'densenet169':
        ftr_num = 1664
    elif backbone_name == 'densenet201':
        ftr_num = 1920
    elif backbone_name == 'nasnetalarge':
        ftr_num = 4032
    elif backbone_name in ['inceptionresnetv2', 'inceptionv4']:
        ftr_num = 1536
    elif backbone_name in ['dpn98', 'dpn92', 'dpn107', 'dpn131']:
        ftr_num = 2688
    elif backbone_name in ['bninception']:
        ftr_num = 1024
    else:
        ftr_num = 2048  # xception, res50, etc...

    return ftr_num

def create_imagenet_backbone(backbone_name, pretrained=True):
    if backbone_name in [
        'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'senet154', 'se_resnet101',
        'se_resnet152', 'nasnetmobile', 'mobilenet', 'nasnetalarge', 'dpn98']:
        backbone = eval(backbone_name)()
    elif backbone_name in ['resnet34', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']:
        backbone = eval(backbone_name)(pretrained=pretrained)
    else:
        raise ValueError('unsupported backbone name {}'.format(backbone_name))
    return backbone
#c = nn.CrossEntropyLoss(reduction='none')
class LandmarkNet(nn.Module):
    def __init__(self, backbone_name, num_classes=1000, start_index=0, pretrained=True, suffix_name='LandmarkNet'):
        super(LandmarkNet, self).__init__()
        print('num_classes:', num_classes)
        self.backbone = create_imagenet_backbone(backbone_name)
        ftr_num = get_num_features(backbone_name)

        self.ftr_num = ftr_num
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(ftr_num, num_classes)
        self.name = '{}_{}_{}_{}'.format(suffix_name, backbone_name, start_index, num_classes)

    def logits(self, x):
        x = self.avg_pool(x)
        x = F.dropout2d(x, 0.4, self.training)
        x = x.view(x.size(0), -1)
        return self.logit(x)#, c(x, label)
    
    def forward(self, x):
        x = self.backbone.features(x)
        return self.logits(x) #, c(x, label)

class FeatureNet(nn.Module):
    def __init__(self, backbone_name, cls_model=None, suffix_name='FeatureNet'):
        super(FeatureNet, self).__init__()
        if cls_model is None:
            self.backbone = create_imagenet_backbone(backbone_name)
        else:
            self.backbone = cls_model.backbone
        self.num_features = get_num_features(backbone_name)
        self.name = '{}_{}'.format(suffix_name, backbone_name)

    def forward(self, x):
        feat = self.backbone.features(x)
        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2, training=self.training)
        #global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        #print(global_feat.size())

        return global_feat

class FeatureNetV1(nn.Module):
    def __init__(self, backbone_name, cls_model=None, suffix_name='FeatureNetV1'):
        super(FeatureNetV1, self).__init__()
        if cls_model is None:
            self.backbone = create_imagenet_backbone(backbone_name)
        else:
            self.backbone = cls_model.backbone
        self.num_features = get_num_features(backbone_name)
        self.name = '{}_{}'.format(suffix_name, backbone_name)
        self.features = nn.Sequential(*(list(self.backbone.children())[:-2]))

    def forward(self, x):
        feat = self.features(x)
        # global feat
        global_feat = LF.rmac(feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        #global_feat = F.dropout(global_feat, p=0.2, training=self.training)
        #global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        #print(global_feat.size())

        return global_feat

class FeatureNetV2(nn.Module):
    def __init__(self, backbone_name, num_classes=14952, cls_model=None, suffix_name='FeatureNetV2'):
        super(FeatureNetV2, self).__init__()
        if cls_model is None:
            self.backbone = create_imagenet_backbone(backbone_name)
        else:
            self.backbone = cls_model.backbone
        self.num_features = get_num_features(backbone_name)
        self.name = '{}_{}'.format(suffix_name, backbone_name)

        local_planes = 512
        self.local_conv = nn.Conv2d(self.num_features, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(self.num_features)
        self.bottleneck_g.bias.requires_grad_(False) 

        self.fc = nn.Linear(self.num_features, num_classes)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        feat = self.backbone.features(x)
        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2, training=self.training)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        #print('global:', global_feat.size())
        # local feat
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        #print('local:', local_feat.size())

        out = self.fc(global_feat) * 16
        #print('out:', out.max(), out.mean(), out.min(), out.std())
        return global_feat, local_feat, out

    def freeze_bn(self):
        for m in self.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()

    def freeze(self):
        for param in self.basemodel.parameters():
            param.requires_grad = False
        if self.model_name.find('dpn98') > -1:
            for param in self.basemodel.features[10:].parameters():
                param.requires_grad = True
        elif self.model_name.find('res') > -1 or self.model_name.find('senet154') > -1:
            for param in self.basemodel.layer3.parameters():
                param.requires_grad = True
            for param in self.basemodel.layer4.parameters():
                param.requires_grad = True
        elif self.model_name.find('inceptionv4') > -1:
            for param in self.basemodel.features[11:].parameters():
                param.requires_grad = True
        elif self.model_name.find('dense') > -1:
            for param in self.basemodel.features[8:].parameters():
                param.requires_grad = True

def create_model(args):
    suffix_name = args.suffix_name #'LandmarkNet'
    #if args.balanced:
    #    suffix_name = 'LandmarkNetB'
    if args.init_ckp is not None:
        model = LandmarkNet(backbone_name=args.backbone, num_classes=args.init_num_classes, start_index=args.start_index, suffix_name=suffix_name)
        model.load_state_dict(torch.load(args.init_ckp))
        if args.init_num_classes != args.num_classes:
            model.logit = nn.Linear(model.ftr_num, args.num_classes)
            model.name = '{}_{}_{}'.format(suffix_name, args.backbone, args.num_classes)
    else:
        model = LandmarkNet(backbone_name=args.backbone, num_classes=args.num_classes, start_index=args.start_index, suffix_name=suffix_name)

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
    x = torch.randn(2, 3, 224, 224).cuda()
    args = Namespace()
    args.init_ckp = None
    args.backbone = 'se_resnet50'
    args.ckp_name = 'best_model.pth'
    args.predict = False
    args.num_classes = 1000

    model = create_model(args)[0].cuda()
    y = model(x)
    print(y.size(), y)

def test_feature_net():
    x = torch.randn(2, 3, 224, 224).cuda()
    model = FeatureNetV1('se_resnet101')
    model.cuda()
    g = model(x)
    #print(y.size())
    print(g.size())

if __name__ == '__main__':
    #test()
    #convert_model4()
    test_feature_net()
