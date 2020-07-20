"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import os
import torch
import models.networks as networks
import util.util as util
import math
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pdb


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE, self.netIG, self.netFE, self.netB, self.netD2, self.netSIG = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionGANFeat = networks.GANFeatLoss(self.opt)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if not opt.no_orient_loss:
                self.criterionOrient = networks.L1OLoss(self.opt)
            
            self.criterionStyleContent = networks.StyleContentLoss(opt)
            # the loss of RGB background
            self.criterionBackground = networks.RGBBackgroundL1Loss()

            self.criterionRGBL1 = nn.L1Loss()
            self.criterionRGBL2 = nn.MSELoss()
            self.criterionLabL1 = networks.LabColorLoss(opt)

            if opt.unpairTrain:
                self.criterionHairAvgLab = networks.HairAvgLabLoss(opt)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        if 'ref' in self.opt.inpaint_mode:
            input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise = self.preprocess_input(data)
        elif 'stroke' in self.opt.inpaint_mode:
            input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise, orient_stroke, mask_stroke, orient_rgb_mask = self.preprocess_input(data)

        if mode == 'generator':
            # print(orient_mask.type, real_image.type)
            g_loss, generated = self.compute_generator_loss(
                input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(image_ref)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.use_ig:
                    hair_mask = torch.unsqueeze(input_tag[:, 1, ...], 1)
                    inpainted_orient_rgb, inpainted_orient_mask = self.inpainting_orient(hole, orient_rgb, noise, hair_mask)
                    inpainted_orient_rgb = np.uint8(inpainted_orient_rgb[0,...].permute(1,2,0).cpu().numpy()*255)
                    fake_image, _, blend_image = self.generate_fake(input_ref, image_ref,
                                                       orient_mask=inpainted_orient_mask, input_tag=input_tag, image_tag=image_tag, noise=noise)
                else:
                    fake_image, _, blend_image = self.generate_fake(input_ref, image_ref,
                                                       orient_mask=orient_mask, input_tag=input_tag, image_tag=image_tag, noise=noise)
            if self.opt.use_blender:
                return blend_image
            return fake_image
        elif mode == 'demo_inference':
            with torch.no_grad():
                if self.opt.use_ig:
                    if 'ref' in self.opt.inpaint_mode:
                        hair_mask = torch.unsqueeze(input_tag[:, 1, ...], 1)
                        inpainted_orient_rgb, inpainted_orient_mask = self.inpainting_orient(hole, orient_rgb, noise, hair_mask)
                        inpainted_orient_rgb = np.uint8(inpainted_orient_rgb[0,...].permute(1,2,0).cpu().numpy()*255)
                        orient_out = inpainted_orient_rgb.copy()
                        fake_image, _, blend_image = self.generate_fake(input_ref, image_ref,
                                                           orient_mask=inpainted_orient_mask, input_tag=input_tag, image_tag=image_tag, noise=noise)
                    elif 'stroke' in self.opt.inpaint_mode:
                        # for stroke orient inpainting
                        hair_mask = torch.unsqueeze(input_tag[:, 1, ...], 1)
                        inpainted_orient_rgb, inpainted_orient_mask = self.inpainting_stroke_orient(hole, orient_rgb, noise, hair_mask, orient_stroke, mask_stroke, orient_rgb_mask)
                        inpainted_orient_rgb = np.uint8(inpainted_orient_rgb[0,...].permute(1,2,0).cpu().numpy()*255)
                        orient_out = inpainted_orient_rgb.copy()
                        fake_image, _, blend_image = self.generate_fake(input_ref, image_ref,
                                                                        orient_mask=inpainted_orient_mask,
                                                                        input_tag=input_tag, image_tag=image_tag,
                                                                        noise=noise)
                else:
                    orient_out = None
                    fake_image, _, blend_image = self.generate_fake(input_ref, image_ref,
                                                       orient_mask=orient_mask, input_tag=input_tag, image_tag=image_tag, noise=noise)
            if self.opt.use_blender:
                return blend_image, orient_out
            return fake_image, orient_out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = []
        if not opt.fix_netG:
            G_params += list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.use_instance_feat:
            G_params += list(self.netFE.parameters())
        if opt.use_blender:
            G_params += list(self.netB.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        if opt.isTrain and opt.unpairTrain:
            D2_params = list(self.netD2.parameters())
            optimizer_D2 = torch.optim.Adam(D2_params, lr=D_lr, betas=(beta1, beta2))
            return optimizer_G, optimizer_D, optimizer_D2
        else:
            return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)
        if self.opt.use_blender:
            util.save_network(self.netB, 'B', epoch, self.opt)
        if self.opt.unpairTrain:
            util.save_network(self.netD2, 'D2', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netD2 = networks.define_D(opt) if opt.isTrain and opt.unpairTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None   # this is for original spade network
        netIG = networks.define_IG(opt) if opt.use_ig else None  # this is the orient inpainting network
        netSIG = networks.define_SIG(opt) if opt.use_stroke else None # this is the stroke orient inpainting network
        netFE = networks.define_FE(opt) if opt.use_instance_feat else None # this is the feat encoder from pix2pixHD
        netB = networks.define_B(opt) if opt.use_blender else None

        if not opt.isTrain or opt.continue_train:
            # if the pth exist
            save_filename = '%s_net_%s.pth' % (opt.which_epoch, 'G')
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            G_path = os.path.join(save_dir, save_filename)
            if os.path.exists(G_path):

                netG = util.load_network(netG, 'G', opt.which_epoch, opt)
                if opt.fix_netG:
                    netG.eval()
                if opt.use_blender:
                    netB = util.load_blend_network(netB, 'B', opt.which_epoch, opt)
                if opt.isTrain:
                    netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                    if opt.unpairTrain:
                        netD2 = util.load_network(netD2, 'D', opt.which_epoch, opt)
                if opt.use_vae:
                    netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        if opt.use_ig:
            netIG = util.load_inpainting_network(netIG, opt)
            netIG.eval()
        if opt.use_stroke:
            netSIG = util.load_sinpainting_network(netSIG, opt)
            netSIG.eval()

        return netG, netD, netE, netIG, netFE, netB, netD2, netSIG

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label_ref'] = data['label_ref'].long()
        data['label_tag'] = data['label_tag'].long()
        if self.use_gpu():
            data['label_ref'] = data['label_ref'].cuda()
            data['label_tag'] = data['label_tag'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image_ref'] = data['image_ref'].cuda()
            data['image_tag'] = data['image_tag'].cuda()
            data['orient'] = data['orient'].cuda()
            data['hole'] = data['hole'].cuda()
            data['orient_rgb'] = data['orient_rgb'].cuda()
            data['noise'] = data['noise'].cuda()
            if 'orient_stroke' in data.keys():
                data['orient_stroke'] = data['orient_stroke'].cuda()
                data['mask_stroke'] = data['mask_stroke'].cuda()
                data['orient_rgb_mask'] = data['orient_rgb_mask'].cuda()


        # create one-hot label map for ref
        label_map = data['label_ref']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_ref = input_label.scatter_(1, label_map, 1.0)

        # create one-hot label map for tag
        label_map_tag = data['label_tag']
        bs, _, h, w = label_map_tag.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_tag = input_label.scatter_(1, label_map_tag, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_tag = torch.cat((input_tag, instance_edge_map), dim=1)

        if 'stroke' in self.opt.inpaint_mode:
            return input_ref, input_tag, data['image_ref'], data['image_tag'], data['orient'], data['hole'], data['orient_rgb'], data['noise'], data['orient_stroke'], data['mask_stroke'],data['orient_rgb_mask']
        else:
            return input_ref, input_tag, data['image_ref'], data['image_tag'], data['orient'], data['hole'], data['orient_rgb'], data['noise']


    def compute_generator_loss(self, input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise):
        G_losses = {}

        if self.opt.use_ig:
            with torch.no_grad():
                hair_mask = torch.unsqueeze(input_tag[:,1,...], 1)
                inpainted_orient_rgb, orient_mask = self.inpainting_orient(hole, orient_rgb, noise, hair_mask)
                orient_mask = orient_mask.detach()
                orient_mask.requires_grad_()

        fake_image, KLD_loss, blend_image = self.generate_fake(
            input_ref, image_ref, compute_kld_loss=self.opt.use_vae, orient_mask=orient_mask, input_tag=input_tag, image_tag=image_tag, noise=noise)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        if self.opt.use_blender:
            pred_fake, pred_real = self.discriminate(
                input_tag, blend_image, image_tag, orient_mask)
        else:
            pred_fake, pred_real = self.discriminate(
                input_tag, fake_image, image_tag, orient_mask)

        # cal GAN loss for generator
        if not self.opt.no_gan_loss:
            label_tag = torch.unsqueeze(input_tag[:, 1, :, :], 1)
            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False, label=label_tag)

        ref_tag_diff = torch.sum(input_tag[:,1,:,:]-input_ref[:,1,:,:])
        ref_tag_diff2 = torch.sum(image_tag - image_ref)
        if ref_tag_diff == 0:
            ref_is_tag = True
        else:
            ref_is_tag = False
        if self.opt.curr_step == 1:
            # cal GAN feat loss
            if not self.opt.no_ganFeat_loss:
                if ref_is_tag:
                    label_tag = torch.unsqueeze(input_tag[:, 1, :, :], 1)
                    G_losses['GAN_Feat'] = self.criterionGANFeat(pred_fake, pred_real, label_tag)


            # cal vgg loss
            if not self.opt.no_vgg_loss:
                if ref_is_tag:
                    label_tag = torch.unsqueeze(input_tag[:, 1, :, :], 1)
                    if self.opt.use_blender:
                        G_losses['VGG'] = self.criterionVGG(blend_image, image_tag, label_tag) * self.opt.lambda_vgg
                    else:
                        G_losses['VGG'] = self.criterionVGG(fake_image, image_tag, label_tag) * self.opt.lambda_vgg

            # cal style and content loss
            style_label = torch.unsqueeze(input_ref[:, 1, :, :], 1)
            content_label = torch.unsqueeze(input_tag[:, 1, :, :], 1)
            if self.opt.use_blender:
                loss_c, loss_s = self.criterionStyleContent(blend_image, image_ref, image_tag, style_label, content_label)
            else:
                loss_c, loss_s = self.criterionStyleContent(fake_image, image_ref, image_tag, style_label, content_label)
            if not self.opt.no_content_loss:
                G_losses['content'] = loss_c * self.opt.lambda_content
            if not self.opt.no_style_loss:
                G_losses['style'] = loss_s * self.opt.lambda_style

            if not self.opt.no_background_loss and ref_is_tag:
                if self.opt.use_blender:
                    background_loss = self.criterionBackground(blend_image, input_tag, image_tag)
                else:
                    background_loss = self.criterionBackground(fake_image, input_tag, image_tag)
                G_losses['background'] = background_loss * self.opt.lambda_background

            if not self.opt.no_rgb_loss and ref_is_tag:
                if self.opt.use_blender:
                    rgb_loss = self.criterionRGBL1(blend_image, image_tag.detach())
                else:
                    rgb_loss = self.criterionRGBL1(fake_image, image_tag.detach())
                G_losses['rgb'] = rgb_loss * self.opt.lambda_rgb

            if not self.opt.no_lab_loss and ref_is_tag:
                if self.opt.use_blender:
                    lab_loss = self.criterionLabL1(blend_image, image_tag.detach(), torch.unsqueeze(input_tag[:, 1, :, :], 1))
                else:
                    lab_loss = self.criterionLabL1(fake_image, image_tag.detach(), torch.unsqueeze(input_tag[:, 1, :, :], 1))
                G_losses['lab'] = lab_loss * self.opt.lambda_lab

        # cal orient loss
        if not self.opt.no_orient_loss:
            if self.opt.use_blender:
                G_losses['ORIENT'], confidence_loss = self.criterionOrient(blend_image, orient_mask, input_tag)
            else:
                G_losses['ORIENT'], confidence_loss = self.criterionOrient(fake_image, orient_mask, input_tag)
            G_losses['ORIENT'] = G_losses['ORIENT'] * self.opt.lambda_orient
            if not self.opt.no_confidence_loss:
                G_losses['CONFIDENCE'] = confidence_loss * self.opt.lambda_confidence

        if self.opt.unpairTrain and self.opt.curr_step == 2:
            if self.opt.use_blender:
                hair_avg_lab_loss = self.criterionHairAvgLab(blend_image, fake_image.detach(), torch.unsqueeze(input_tag[:,1,:,:], 1), torch.unsqueeze(input_tag[:,1,:,:], 1))
            else:
                hair_avg_lab_loss = self.criterionHairAvgLab(fake_image, image_ref, torch.unsqueeze(input_tag[:, 1, :, :], 1),
                                                             torch.unsqueeze(input_ref[:,1,:,:], 1))
            G_losses['hairAvgLab'] = hair_avg_lab_loss * self.opt.lambda_hairavglab
            if self.opt.use_blender:
                background_loss = self.criterionBackground(blend_image, input_tag, image_tag)
            else:
                background_loss = self.criterionBackground(fake_image, input_tag, image_tag)
            G_losses['background'] = background_loss * self.opt.lambda_background

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_ref, input_tag, image_ref, image_tag, orient_mask, hole, orient_rgb, noise):
        D_losses = {}
        with torch.no_grad():
            if self.opt.use_ig:
                hair_mask = torch.unsqueeze(input_tag[:, 1, ...], 1)
                inpainted_orient_rgb, orient_mask = self.inpainting_orient(hole, orient_rgb, noise, hair_mask)
                orient_mask = orient_mask.detach()
                orient_mask.requires_grad_()

            fake_image, _, blend_image = self.generate_fake(input_ref, image_ref, compute_kld_loss=self.opt.use_vae, orient_mask=orient_mask, input_tag=input_tag,
                                               image_tag=image_tag, noise=noise)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            if self.opt.use_blender:
                blend_image = blend_image.detach()
                blend_image.requires_grad_()

        if self.opt.use_blender:
            pred_fake, pred_real = self.discriminate(
                input_tag, blend_image, image_tag, orient_mask)
        else:
            pred_fake, pred_real = self.discriminate(
                input_tag, fake_image, image_tag, orient_mask)

        # cal GAN loss for discriminator
        label_tag = torch.unsqueeze(input_tag[:, 1, :, :], 1)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True, label=label_tag)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True, label=label_tag)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar



    def inpainting_orient(self, hole, orient_rgb, noise, mask):
        orient_rgb_hole = orient_rgb * (1 - hole) + noise * hole
        input = torch.cat([orient_rgb_hole, hole], dim=1)
        if self.opt.crop_size != 256:
            input = F.interpolate(input, size=(256, 256), mode='nearest')
        output = self.netIG(input)

        if self.opt.crop_size != 256:
            output = F.interpolate(output, size=(self.opt.crop_size, self.opt.crop_size), mode='nearest')
        output = output * hole + orient_rgb * (1 - hole) #[n, 3, h, w], [0, 1]

        # trans to orient with one channel
        orient = output.clone()
        orient = (orient - 0.5) * 2 # [-1, 1]
        orient = torch.acos(orient[:,0,:,:]) / 2.0 / math.pi * 255.0
        orient = torch.round(torch.unsqueeze(orient, dim=1) * mask)
        # trains to new orient with two channel
        orient2 = (output[:,:-1,:,:]-0.5) * 2
        orient = orient2.clone()
        orient[:, 0, :, :] = orient2[:, 1, :, :]
        orient[:, 1, :, :] = orient2[:, 0, :, :]
        orient = orient * mask
        return output, orient

    def inpainting_stroke_orient(self, hole, orient_rgb, noise, mask, stroke, stroke_mask, mask_orient_rgb):
        if torch.max(mask-mask_orient_rgb) != 0:
            print('inpainting first')
            hole0 = mask-mask_orient_rgb
            orient_rgb_0 = orient_rgb * (1 - hole0) + noise * hole0
            orient_rgb_two_hole = orient_rgb_0 * (1 - hole) + noise * (hole-stroke_mask) + stroke * stroke_mask
            # self.save_orient_image(orient_rgb_two_hole, 'input_orient_two_noise.png')
            orient_rgb_1,_ = self.inpainting_orient(mask-mask_orient_rgb,orient_rgb,noise,mask)

        else:
            orient_rgb_1 = orient_rgb.clone()

        orient_rgb_hole = orient_rgb_1 * (1 - hole) + noise * (hole-stroke_mask) + stroke * stroke_mask
        # self.save_orient_image(orient_rgb_hole, 'input_orient_noise.png')
        input = torch.cat([orient_rgb_hole, hole, stroke_mask], dim=1)
        if self.opt.crop_size != 256:
            input = F.interpolate(input, size=(256, 256), mode='nearest')
        output = self.netSIG(input)

        if self.opt.crop_size != 256:
            output = F.interpolate(output, size=(self.opt.crop_size, self.opt.crop_size), mode='nearest')
        output = output * hole + orient_rgb_1 * (1 - hole) #[n, 3, h, w], [0, 1]
        # trans to orient with one channel
        orient = output.clone()
        orient = (orient - 0.5) * 2 # [-1, 1]
        orient = torch.acos(orient[:,0,:,:]) / 2.0 / math.pi * 255.0
        orient = torch.round(torch.unsqueeze(orient, dim=1) * mask)
        # trains to new orient with two channel
        orient2 = (output[:,:-1,:,:]-0.5) * 2
        orient = orient2.clone()
        orient[:, 0, :, :] = orient2[:, 1, :, :]
        orient[:, 1, :, :] = orient2[:, 0, :, :]
        orient = orient * mask
        return output, orient

    def save_image(self, image,name=None):
        image_numpy = image[0,...].cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
        image_numpy = (image_numpy + 1) / 2 * 255.0
        image_pil = Image.fromarray(np.uint8(image_numpy))
        if name is None:
            image_pil.save('./inference_samples/fake_image_noblend.jpg')
        else:
            image_pil.save('./inference_samples/'+name)

    def save_orient_image(self, image,name=None):
        image_numpy = image[0,...].cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
        image_numpy = image_numpy * 255.0
        image_pil = Image.fromarray(np.uint8(image_numpy))
        image_pil.save('./inference_samples/'+name)

    def save_blend_input(self, image, image_tag):
        image_numpy = image[0,...].cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
        image_numpy = (image_numpy + 1) / 2 * 255.0
        image_pil = Image.fromarray(np.uint8(image_numpy))
        image_pil.save('./inference_samples/blend_image_ref.jpg')
        image_numpy = image_tag[0,...].cpu().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) #[h,w,3]
        image_numpy = (image_numpy + 1) / 2 * 255.0
        image_pil = Image.fromarray(np.uint8(image_numpy))
        image_pil.save('./inference_samples/blend_image_tag.jpg')

    def zeros_padding(self, input):
        N, C, H, W = input.size()
        th = self.opt.add_th
        FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
        pad_output = FloatTensor(np.zeros((N, C, H+th, W+th)))
        # print(input.size(), pad_output.size())
        pad_output[:,:,int(th/2):int(th/2)+H,int(th/2):int(th/2)+W] = input
        return pad_output


    def generate_fake(self, input_ref, image_ref, compute_kld_loss=False, orient_mask=None, input_tag=None, image_tag=None, noise=None):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(image_ref.clone())
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        if self.opt.add_feat_zeros:
            input_ref = self.zeros_padding(input_ref)
            image_ref = self.zeros_padding(image_ref)
            orient_mask = self.zeros_padding(orient_mask)
            input_tag = self.zeros_padding(input_tag)
            image_tag = self.zeros_padding(image_tag)
            noise = self.zeros_padding(noise)

        if not self.opt.only_blend:
            fake_image = self.netG(input_ref, z=z, orient_mask=orient_mask, image_ref=image_ref, input_tag=input_tag, noise=noise, image_tag=image_tag)

        else:
            fake_image = None

        if self.opt.use_blender:
            if self.opt.only_blend:
                blend_image = self.netB(image_ref, image_tag, input_tag, noise)
            else:
                blend_image = self.netB(fake_image, image_tag, input_tag, noise)
        else:
            blend_image = None

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        # if (fake_image != fake_image).sum() > 0:
        #     pdb.set_trace()

        return fake_image, KLD_loss, blend_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_tag, fake_image, real_image, orient_mask=None):
        # process the orient mask to sin cos mask
        if not self.opt.use_ig:
            orient_mask = orient_mask / 255.0 * math.pi
            orient_input = torch.cat([torch.sin(2 * orient_mask), torch.cos(2 * orient_mask)], dim=1)  # [n,2,h,w]
            orient_input = orient_input * torch.unsqueeze(input_tag[:, 1, :, :], 1)  # hair_mask==seg[:,1,:,:]
        else:
            orient_input = orient_mask
        # # remove the background if specified
        # if self.opt.remove_background:
        #     fake_image = fake_image * torch.unsqueeze(input_tag[:, 1, :, :], 1)
        #     real_image = real_image * torch.unsqueeze(input_tag[:, 1, :, :], 1)
        # stack the input
        fake_concat = torch.cat([input_tag, orient_input, fake_image], dim=1)
        real_concat = torch.cat([input_tag, orient_input, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        if self.opt.unpairTrain:
            if self.opt.curr_step == 1:
                discriminator_out = self.netD(fake_and_real)
            else:
                discriminator_out = self.netD2(fake_and_real)
        else:
            discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
