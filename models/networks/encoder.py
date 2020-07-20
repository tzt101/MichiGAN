"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.MaskGAN_networks import ConvBlock
import torch
from models.networks.partialconv2d import PartialConv2d
import random


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class ImageEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt, sw, sh):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf # 64
        self.sw = sw
        self.sh = sh
        self.opt = opt
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.adaptivepool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.fc = nn.Conv2d(ndf*16, ndf*16*self.sw*self.sh, 1, 1, 0)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x, label_ref=None, label_tag=None):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #    x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        x = self.adaptivepool(x)
        x = self.fc(x)
        x = x.view(x.size()[0], self.opt.ngf*16, self.sh, self.sw)

        return x

class ImageEncoder2(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt, sw, sh):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf # 64
        self.sw = sw
        self.sh = sh
        self.opt = opt
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x, label_ref, label_tag):

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #    x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        # resize the label
        _,_,xh,xw = x.size()
        label_ref = F.interpolate(label_ref, size=(xh, xw), mode='nearest')
        label_tag = F.interpolate(label_tag, size=(xh, xw), mode='nearest')
        # instance_wise(hair) average pool
        outputs_mean = x.clone()
        for b in range(x.size()[0]):
            if self.opt.ref_global_pool:
                tmps = x[b,...]
                tmps = torch.mean(torch.mean(tmps, dim=1, keepdim=True), dim=2, keepdim=True)
            else:
                tmps = x[b, ...] * label_ref[b, ...]
                tmps = torch.sum(torch.sum(tmps, dim=1, keepdim=True), dim=2, keepdim=True) / max(torch.sum(label_ref[b, ...]), 1)
            tmps = tmps.expand_as(x[b, ...])
            outputs_mean[b, ...] = tmps * label_tag[b]
        # resize
        if self.sh != xh:
            outputs_mean = F.interpolate(outputs_mean, size=(self.sh, self.sw), mode='nearest')

        return outputs_mean

class ImageEncoder3(BaseNetwork):
    """ Same architecture as the image discriminator (Partial Convolution)"""

    def __init__(self, opt, sw, sh):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf # 64
        self.sw = sw
        self.sh = sh
        self.opt = opt
        self.layer1 = PartialConv2d(3, ndf, kw, stride=2, padding=pw, return_mask=True)
        self.norm1 = nn.InstanceNorm2d(ndf, affine=False)
        self.layer2 = PartialConv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw, return_mask=True)
        self.norm2 = nn.InstanceNorm2d(ndf * 2, affine=False)
        self.layer3 = PartialConv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw, return_mask=True)
        self.norm3 = nn.InstanceNorm2d(ndf * 4, affine=False)
        self.layer4 = PartialConv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw, return_mask=True)
        self.norm4 = nn.InstanceNorm2d(ndf * 8, affine=False)
        self.layer5 = PartialConv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw, return_mask=True)
        self.norm5 = nn.InstanceNorm2d(ndf * 16, affine=False)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x, label_ref0, label_tag0):

        if 'instance' in self.opt.norm_ref_encode:

            x, mask = self.layer1(x, label_ref0)
            x = self.norm1(x)
            x, mask = self.layer2(self.actvn(x), mask)
            x = self.norm2(x)
            x, mask = self.layer3(self.actvn(x), mask)
            x = self.norm3(x)
            x, mask = self.layer4(self.actvn(x), mask)
            x = self.norm4(x)
            x, mask = self.layer5(self.actvn(x), mask)
            x = self.norm5(x)
        elif 'none' in self.opt.norm_ref_encode:
            x, mask = self.layer1(x, label_ref0)
            x, mask = self.layer2(self.actvn(x), mask)
            x, mask = self.layer3(self.actvn(x), mask)
            x, mask = self.layer4(self.actvn(x), mask)
            x, mask = self.layer5(self.actvn(x), mask)

        x = self.actvn(x)
        # print('save feat')
        # self.show_feature_map(x, min_channel=0, max_channel=25)
        # resize the label
        _,_,xh,xw = x.size()
        label_ref = F.interpolate(label_ref0, size=(xh, xw), mode='nearest')
        label_tag = F.interpolate(label_tag0, size=(xh, xw), mode='nearest')
        # instance_wise(hair) average pool
        outputs_mean = x.clone()
        for b in range(x.size()[0]):
            tmps = x[b, ...] * label_ref[b, ...]
            tmps = torch.sum(torch.sum(tmps, dim=1, keepdim=True), dim=2, keepdim=True) / max(torch.sum(label_ref[b, ...]), 1)
            tmps = tmps.expand_as(x[b, ...])
            outputs_mean[b, ...] = tmps * label_tag[b]
        # resize
        if self.sh != xh:
            outputs_mean = F.interpolate(outputs_mean, size=(self.sh, self.sw), mode='bilinear')

        return outputs_mean

class BackgroundEncode(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        self.conv1 = ConvBlock(3, self.ngf, 7, 1, 3, norm='none', activation='relu', pad_type='reflect')
        self.layer1 = ConvBlock(self.ngf, 2 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer2 = ConvBlock(2 * self.ngf, 4 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer3 = ConvBlock(4 * self.ngf, 8 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer4 = ConvBlock(8 * self.ngf, 16 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')

    def forward(self, image, mask):
        x0 = self.conv1(image) # 64
        x1 = self.layer1(x0) # 1/2 64*2
        x2 = self.layer2(x1) # 1/4 64*4
        x3 = self.layer3(x2) # 1/8 64*8
        x4 = self.layer4(x3) # 1/16 64*16

        back_mask = torch.unsqueeze(mask[:,0,:,:], 1)
        _,_,sh,sw = back_mask.size()
        back_mask1 = F.interpolate(back_mask, size=(int(sh/2), int(sw/2)), mode='nearest')
        back_mask2 = F.interpolate(back_mask, size=(int(sh / 4), int(sw / 4)), mode='nearest')
        back_mask3 = F.interpolate(back_mask, size=(int(sh / 8), int(sw / 8)), mode='nearest')
        back_mask4 = F.interpolate(back_mask, size=(int(sh / 16), int(sw / 16)), mode='nearest')


        return [x0, x1, x2, x3, x4], [back_mask, back_mask1, back_mask2, back_mask3, back_mask4]

def save_image(image, name):
    from PIL import Image
    image_numpy = image[0,...].cpu().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
    image_numpy = (image_numpy + 1) / 2 * 255.0
    image_pil = Image.fromarray(np.uint8(image_numpy))
    image_pil.save('./inference_samples/'+name)

def save_mask(mask, name):
    from PIL import Image
    image_numpy = mask[0,0,...].cpu().numpy()
    image_numpy = image_numpy * 255.0
    image_pil = Image.fromarray(np.uint8(image_numpy))
    image_pil.save('./inference_samples/'+name)


class BackgroundEncode2(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ngf = opt.ngf
        if opt.num_upsampling_layers == 'most':
            self.conv0 = ConvBlock(3, self.ngf // 2, 7, 1, 3, norm='none', activation='relu', pad_type='reflect')
            self.layer0 = ConvBlock(self.ngf // 2, self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        else:
            self.conv1 = ConvBlock(3, self.ngf, 7, 1, 3, norm='none', activation='relu', pad_type='reflect')
        self.layer1 = ConvBlock(self.ngf, 2 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer2 = ConvBlock(2 * self.ngf, 4 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer3 = ConvBlock(4 * self.ngf, 8 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')
        self.layer4 = ConvBlock(8 * self.ngf, 16 * self.ngf, 4, 2, 1, norm='none', activation='relu', pad_type='reflect')

    def forward(self, image, mask, noise):

        if self.opt.isTrain:
            if self.opt.random_expand_mask:
                hair_mask = torch.unsqueeze(mask[:, 1, :, :], 1)
                _,_,mh,mw = hair_mask.shape
                th = int(mh * self.opt.random_expand_th)
                th = th if th % 2 == 1 else th+1
                k = random.choice([max(th-4,1),max(th-2,1),th,th+2,th+4])
                p = int(k / 2)
                expand_hair_mask = F.max_pool2d(hair_mask, kernel_size=k, stride=1, padding=p)
                back_mask = 1 - expand_hair_mask
            else:
                back_mask = torch.unsqueeze(mask[:, 0, :, :], 1)
        else:
            if self.opt.expand_mask_be:
                hair_mask = torch.unsqueeze(mask[:, 1, :, :], 1)
                k = self.opt.expand_th
                p = int(k / 2)
                if self.opt.add_feat_zeros:
                    th = self.opt.add_th
                    H, W = self.opt.crop_size, self.opt.crop_size
                    expand_hair_mask = hair_mask * 0
                    hair_no_pad = hair_mask[:,:,int(th/2):int(th/2)+H,int(th/2):int(th/2)+W]
                    expand_hair_no_pad = F.max_pool2d(hair_no_pad, kernel_size=k, stride=1, padding=p)
                    expand_hair_mask[:,:,int(th/2):int(th/2)+H,int(th/2):int(th/2)+W] = expand_hair_no_pad
                else:
                    expand_hair_mask = F.max_pool2d(hair_mask, kernel_size=k, stride=1, padding=p)
                back_mask = 1 - expand_hair_mask
            else:
                back_mask = torch.unsqueeze(mask[:, 0, :, :], 1)

        if self.opt.random_noise_background:
            input = noise
        else:
            input = image * back_mask + noise * (1 - back_mask)

        if self.opt.num_upsampling_layers == 'most':
            x00 = self.conv0(input) # 64 *0.5
            x0 = self.layer0(x00) # 64 1/2
        else:
            x0 = self.conv1(input) # 64
        x1 = self.layer1(x0) # 1/2 64*2
        x2 = self.layer2(x1) # 1/4 64*4
        x3 = self.layer3(x2) # 1/8 64*8

        _,_,sh,sw = back_mask.size()
        back_mask1 = F.interpolate(back_mask, size=(int(sh/2), int(sw/2)), mode='nearest')
        back_mask2 = F.interpolate(back_mask, size=(int(sh / 4), int(sw / 4)), mode='nearest')
        back_mask3 = F.interpolate(back_mask, size=(int(sh / 8), int(sw / 8)), mode='nearest')
        back_mask4 = F.interpolate(back_mask, size=(int(sh / 16), int(sw / 16)), mode='nearest')

        if self.opt.num_upsampling_layers == 'most':
            return [x3, x2, x1, x0, x00], [back_mask4, back_mask3, back_mask2, back_mask1, back_mask]
        else:
            return [x3, x2, x1, x0], [back_mask3, back_mask2, back_mask1, back_mask]



