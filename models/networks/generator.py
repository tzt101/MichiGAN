"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock1
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import SPADEImageBlock as SPADEImageBlock
from models.networks.encoder import ImageEncoder, ImageEncoder2, BackgroundEncode, BackgroundEncode2, ImageEncoder3
import math
from models.networks.MaskGAN_networks import StyleEncoder, LabelEncoder, ResnetBlock2

class SPADEBGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        elif opt.use_encoder:
            # In case of encoder, we will encoder the image
            if self.opt.Image_encoder_mode == 'norm':
                self.fc = ImageEncoder(opt, self.sw, self.sh)
            elif self.opt.Image_encoder_mode == 'instance':
                self.fc = ImageEncoder2(opt, self.sw, self.sh)
            elif self.opt.Image_encoder_mode == 'partialconv':
                self.fc = ImageEncoder3(opt, self.sw, self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            if not opt.no_orientation:
                # self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1) # for mask input
                self.fc = nn.Conv2d(3, 16 * nf, 3, padding=1) # for image input
            else:
                # self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1) # for mask input
                self.fc = nn.Conv2d(3, 16 * nf, 3, padding=1) # for image input

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        if not self.opt.noise_background:
            self.backgroud_enc = BackgroundEncode(opt)
        else:
            self.backgroud_enc = BackgroundEncode2(opt)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        if opt.add_feat_zeros:
            sw = (opt.crop_size+opt.add_th) // (2**num_up_layers)
        else:
            sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def get_wide_edges(self, t):
        n, c, h, w = t.size()
        k = 5
        p = int(k / 2)
        out = 1 - F.max_pool2d(1 - t, kernel_size=k, stride=1, padding=p)
        edges = t - out
        edges = F.interpolate(edges, size=(h, w), mode='nearest')
        return edges

    def forward(self, input=None, z=None, orient_mask=None, image_ref=None, input_tag=None, noise=None, image_tag = None):
        seg = input_tag

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input_tag.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input_tag.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        elif self.opt.use_encoder:
            if self.opt.Image_encoder_mode == 'norm':
                x = self.fc(image_ref)
            elif self.opt.Image_encoder_mode == 'instance' or self.opt.Image_encoder_mode == 'partialconv':
                ins_ref = torch.unsqueeze(input[:, 1, :, :], dim=1)
                ins_tag = torch.unsqueeze(input_tag[:, 1, :, :], dim=1)
                x = self.fc(image_ref, ins_ref, ins_tag)
        else:
            # we downsample segmap (or image without hair) and run convolution
            x = F.interpolate(image_ref, size=(self.sh, self.sw))
            x = self.fc(x)

        if not self.opt.no_orientation:
            if not self.opt.use_ig:
                orient_mask1 = orient_mask / 255.0 * math.pi
                orient_input = torch.cat([torch.sin(2 * orient_mask1), torch.cos(2 * orient_mask1)], dim=1)  # [n,2,h,w]
                orient_input = orient_input *torch.unsqueeze(seg[:, 1, :, :], 1)  # hair_mask==seg[:,1,:,:]
            else:
                orient_input = orient_mask
            # process orient to random
            if self.opt.orient_random_disturb:
                edges = self.get_wide_edges(torch.unsqueeze(input_tag[:, 1, :, :], 1))
                random_value = edges * noise[:,:1,:,:]
                orient_input = orient_input * (1 - edges) + random_value

            seg = torch.cat([seg, orient_input], dim=1)

        if not self.opt.noise_background:
            back_feats, back_masks = self.backgroud_enc(image_tag, input_tag)
        else:
            back_feats, back_masks = self.backgroud_enc(image_tag, input_tag, noise)

        hair_mask = torch.unsqueeze(input_tag[:, 1, :, :], dim=1)
        _,_,sh,sw = hair_mask.size()
        hair_mask1 = F.interpolate(hair_mask, size=(int(sh/2), int(sw/2)), mode='nearest')
        hair_mask2 = F.interpolate(hair_mask, size=(int(sh / 4), int(sw / 4)), mode='nearest')
        hair_mask3 = F.interpolate(hair_mask, size=(int(sh / 8), int(sw / 8)), mode='nearest')
        hair_mask4 = F.interpolate(hair_mask, size=(int(sh / 16), int(sw / 16)), mode='nearest')

        if self.opt.num_upsampling_layers == 'most':
            hair_masks = [hair_mask4, hair_mask3, hair_mask2, hair_mask1, hair_mask]
        else:
            hair_masks = [hair_mask3, hair_mask2, hair_mask1, hair_mask]

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)


        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        if self.opt.use_clip:
            x[x > self.opt.clip_th] = 0
            print('G_middle_1:', x.max())

        x = self.up(x)
        x = self.up_0(x, seg)

        if self.opt.use_clip:
            x[x > self.opt.clip_th] = 0
            print('up_0:', x.max())
        if self.opt.bf_direct_add:
            x = back_feats[0] + x
        else:
            x = back_feats[0] * (1 - hair_masks[0]) + x * (1 - back_masks[0])

        x = self.up(x)
        x = self.up_1(x, seg)

        if self.opt.use_clip:
            x[x > self.opt.clip_th] = 0
            print('up_1:', x.max())
        if self.opt.bf_direct_add:
            x = back_feats[1] + x
        else:
            x = back_feats[1] * (1 - hair_masks[1]) + x * (1 - back_masks[1])

        x = self.up(x)
        x = self.up_2(x, seg)

        if self.opt.use_clip:
            x[x > self.opt.clip_th] = 0
            print('up_2:', x.max())
        if self.opt.bf_direct_add:
            x = back_feats[2] + x
        else:
            x = back_feats[2] * (1 - hair_masks[2]) + x * (1 - back_masks[2])

        x = self.up(x)
        x = self.up_3(x, seg) # 64

        if self.opt.use_clip:
            x[x > self.opt.clip_th] = 0
            print('up_3:', x.max())
        if self.opt.bf_direct_add:
            x = back_feats[3] + x
        else:
            x = back_feats[3] * (1 - hair_masks[3]) + x * (1 - back_masks[3])

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
            x = back_feats[4] * (1 - hair_masks[4]) + x * (1 - back_masks[4])


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


############################################### vgg unet define #############################################
def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


def upsample(in_features, out_features):
    shape = out_features.shape[2:]  # h w
    return F.upsample(in_features, size=shape, mode='bilinear', align_corners=True)


def concat(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)

############################################# Residual Conv ###################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_model='instance'):
        super(BasicBlock, self).__init__()
        if 'instance' == norm_model:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    def __init__(self, opt, ngf=32, norm_layer=nn.InstanceNorm2d,
                     padding_type='reflect'):
        super(Blend2Generator, self).__init__()
        activation = nn.LeakyReLU(0.2, True)
        self.opt = opt
        input_nc = 4
        output_nc = 3

        model1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                     activation]
        self.model1 = nn.Sequential(*model1)
        ### downsample
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model2 = nn.Sequential(*model2)
        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model3 = nn.Sequential(*model3)
        mult = 2 ** 2
        model4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model4 = nn.Sequential(*model4)


        ### resnet blocks
        mult = 2 ** 3
        model_middle = []
        for i in range(3):
            model_middle += [ResnetBlock2(ngf * mult, norm_type='in', padding_type=padding_type)]
        self.model_middle = nn.Sequential(*model_middle)

        ### upsample
        mult = 2 ** 3
        model5 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                          norm_layer(int(ngf * mult / 2)), activation]
        self.model5 = nn.Sequential(*model5)
        mult = 2 ** 2
        model6 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                          norm_layer(int(ngf * mult / 2)), activation]
        self.model6 = nn.Sequential(*model6)
        mult = 2 ** 1
        model7 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                          norm_layer(int(ngf * mult / 2)), activation]
        self.model7 = nn.Sequential(*model7)


        model8 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0)]
        self.model8 = nn.Sequential(*model8)

    def save_image(self, image):
        import numpy as np
        from PIL import Image
        image_numpy = image[0,...].cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
        image_numpy = (image_numpy + 1) / 2 * 255.0
        image_pil = Image.fromarray(np.uint8(image_numpy))
        image_pil.save('./inference_samples/blend_input.jpg')

    def get_wide_edges(self, t):
        n, c, h, w = t.size()
        k = 5
        p = int(k / 2)
        out = F.max_pool2d(t, kernel_size=k, stride=1, padding=p)
        out2 = 1 - F.max_pool2d(1 - t, kernel_size=k, stride=1, padding=p)
        edges = out - out2
        edges = F.interpolate(edges, size=(h, w), mode='nearest')
        return edges

    def forward(self, hair, background, mask, noise=None):
        # mask is one-hot mode
        n, c, h, w = mask.size()
        if c > 1:
            hair_mask = torch.unsqueeze(mask[:, 1, :, :], 1)
        else:
            hair_mask = mask
        input = hair * hair_mask + background * (1 - hair_mask)

        if not self.opt.isTrain:
            self.save_image(input)

        if self.opt.hair_random_disturb:
            edgs = self.get_wide_edges(hair_mask)
            input = input * (1 - edgs) + noise * edgs

        e_1 = self.model1(torch.cat([input, hair_mask], 1))
        e_2 = self.model2(e_1)
        e_3 = self.model3(e_2)
        e_4 = self.model4(e_3)
        e_m = self.model_middle(e_4)
        e_5 = self.model5(torch.cat([e_m, e_4], 1))
        e_6 = self.model6(torch.cat([e_5, e_3], 1))
        e_7 = self.model7(torch.cat([e_6, e_2], 1))
        e_8 = self.model8(torch.cat([e_7, e_1], 1))
        return F.tanh(e_8)

    ####################################### inpainting generator ####################################################
class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, stride = 1, padding = 0, dilation = 2)),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, stride = 1, padding = 0)),
            nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.conv_block(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, downsample = 4):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels = dim, out_channels = dim // downsample, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = dim, out_channels = dim // downsample, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        batch_size, depth, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, depth, width, height)
        return torch.cat([x, out], dim = 1)


class InpaintGenerator(BaseNetwork):

    def __init__(self, opt, blocks = 12, skips = False):
        super().__init__()

        self.skips = skips

        if skips:
            encoder_1 = []
            encoder_1.append(nn.ReflectionPad2d(3))
            encoder_1.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 0)))
            encoder_1.append(nn.InstanceNorm2d(64))
            encoder_1.append(nn.LeakyReLU(0.2, True))
            self.encoder_1 = nn.Sequential(*encoder_1)
            encoder_2 = []
            encoder_2.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            encoder_2.append(nn.InstanceNorm2d(128))
            encoder_2.append(nn.LeakyReLU(0.2, True))
            self.encoder_2 = nn.Sequential(*encoder_2)
            encoder_3 = []
            encoder_3.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)))
            encoder_3.append(nn.InstanceNorm2d(256))
            encoder_3.append(nn.LeakyReLU(0.2, True))
            self.encoder_3 = nn.Sequential(*encoder_3)
        else:
            encoder = []
            encoder.append(nn.ReflectionPad2d(3))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 7, padding = 0)))
            encoder.append(nn.InstanceNorm2d(64))
            encoder.append(nn.LeakyReLU(0.2, True))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            encoder.append(nn.InstanceNorm2d(128))
            encoder.append(nn.LeakyReLU(0.2, True))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)))
            encoder.append(nn.InstanceNorm2d(256))
            encoder.append(nn.LeakyReLU(0.2, True))
            self.encoder = nn.Sequential(*encoder)

        middle = []
        for _ in range(blocks):
            middle.append(ResnetBlock(256))
        middle.append(SelfAttention(256))
        self.middle = nn.Sequential(*middle)

        if skips:
            decoder_3 = []
            decoder_3.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 768, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            decoder_3.append(nn.InstanceNorm2d(128))
            decoder_3.append(nn.ReLU(True))
            self.decoder_3 = nn.Sequential(*decoder_3)
            decoder_2 = []
            decoder_2.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)))
            decoder_2.append(nn.InstanceNorm2d(64))
            decoder_2.append(nn.ReLU(True))
            self.decoder_2 = nn.Sequential(*decoder_2)
            decoder_1 = []
            decoder_1.append(nn.ReflectionPad2d(3))
            decoder_1.append(nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size = 7, padding = 0))
            self.decoder_1 = nn.Sequential(*decoder_1)
        else:
            decoder = []
            decoder.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            decoder.append(nn.InstanceNorm2d(128))
            decoder.append(nn.ReLU(True))
            decoder.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)))
            decoder.append(nn.InstanceNorm2d(64))
            decoder.append(nn.ReLU(True))
            decoder.append(nn.ReflectionPad2d(3))
            decoder.append(nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 7, padding = 0))
            self.decoder = nn.Sequential(*decoder)

        # self.init_weights()

    def forward(self, x):
        if self.skips:
            e_1 = self.encoder_1(x)
            e_2 = self.encoder_2(e_1)
            e_3 = self.encoder_3(e_2)
            d_3 = self.middle(e_3)
            d_2 = self.decoder_3(torch.cat([d_3, e_3], dim = 1))
            d_1 = self.decoder_2(torch.cat([d_2, e_2], dim = 1))
            out = self.decoder_1(torch.cat([d_1, e_1], dim = 1))
        else:
            x = self.encoder(x)
            x = self.middle(x)
            out = self.decoder(x)
        out = (torch.tanh(out) + 1) / 2
        return out

class SInpaintGenerator(BaseNetwork):

    def __init__(self, opt, blocks = 12, skips = False):
        super(SInpaintGenerator, self).__init__()

        self.skips = skips
        self.opt = opt

        if skips:
            encoder_1 = []
            encoder_1.append(nn.ReflectionPad2d(3))
            encoder_1.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 5, out_channels = 64, kernel_size = 7, padding = 0)))
            encoder_1.append(nn.InstanceNorm2d(64))
            encoder_1.append(nn.LeakyReLU(0.2, True))
            self.encoder_1 = nn.Sequential(*encoder_1)
            encoder_2 = []
            encoder_2.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            encoder_2.append(nn.InstanceNorm2d(128))
            encoder_2.append(nn.LeakyReLU(0.2, True))
            self.encoder_2 = nn.Sequential(*encoder_2)
            encoder_3 = []
            encoder_3.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)))
            encoder_3.append(nn.InstanceNorm2d(256))
            encoder_3.append(nn.LeakyReLU(0.2, True))
            self.encoder_3 = nn.Sequential(*encoder_3)
        else:
            encoder = []
            encoder.append(nn.ReflectionPad2d(3))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 5, out_channels = 64, kernel_size = 7, padding = 0)))
            encoder.append(nn.InstanceNorm2d(64))
            encoder.append(nn.LeakyReLU(0.2, True))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            encoder.append(nn.InstanceNorm2d(128))
            encoder.append(nn.LeakyReLU(0.2, True))
            encoder.append(nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)))
            encoder.append(nn.InstanceNorm2d(256))
            encoder.append(nn.LeakyReLU(0.2, True))
            self.encoder = nn.Sequential(*encoder)

        middle = []
        for _ in range(blocks):
            middle.append(ResnetBlock(256))
        middle.append(SelfAttention(256))
        self.middle = nn.Sequential(*middle)

        if skips:
            decoder_3 = []
            decoder_3.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 768, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            decoder_3.append(nn.InstanceNorm2d(128))
            decoder_3.append(nn.ReLU(True))
            self.decoder_3 = nn.Sequential(*decoder_3)
            decoder_2 = []
            decoder_2.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)))
            decoder_2.append(nn.InstanceNorm2d(64))
            decoder_2.append(nn.ReLU(True))
            self.decoder_2 = nn.Sequential(*decoder_2)
            decoder_1 = []
            decoder_1.append(nn.ReflectionPad2d(3))
            decoder_1.append(nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size = 7, padding = 0))
            self.decoder_1 = nn.Sequential(*decoder_1)
        else:
            decoder = []
            decoder.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)))
            decoder.append(nn.InstanceNorm2d(128))
            decoder.append(nn.ReLU(True))
            decoder.append(nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)))
            decoder.append(nn.InstanceNorm2d(64))
            decoder.append(nn.ReLU(True))
            decoder.append(nn.ReflectionPad2d(3))
            decoder.append(nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 7, padding = 0))
            self.decoder = nn.Sequential(*decoder)

        self.init_weights()

    def forward(self, x):
        if self.skips:
            e_1 = self.encoder_1(x)
            e_2 = self.encoder_2(e_1)
            e_3 = self.encoder_3(e_2)
            d_3 = self.middle(e_3)
            d_2 = self.decoder_3(torch.cat([d_3, e_3], dim = 1))
            d_1 = self.decoder_2(torch.cat([d_2, e_2], dim = 1))
            out = self.decoder_1(torch.cat([d_1, e_1], dim = 1))
        else:
            x = self.encoder(x)
            x = self.middle(x)
            out = self.decoder(x)
        out = (torch.tanh(out) + 1) / 2
        return out