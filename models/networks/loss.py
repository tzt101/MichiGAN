"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19
import math
import numpy as np
from torch.nn.functional import grid_sample


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def get_wide_edges(self, t, th=0.06):
        n, c, h, w = t.size()
        k = max(1, int(h * th))
        p = int(k / 2)
        out = F.max_pool2d(t, kernel_size=k, stride=1, padding=p)
        out2 = 1 - F.max_pool2d(1 - t, kernel_size=k, stride=1, padding=p)
        edges = out - out2
        edges = F.interpolate(edges, size=(h, w), mode='nearest')
        return edges

    def get_weight_mask(self, input, mask):
        # resize the mask
        n, c, h, w = input.size()
        label = F.interpolate(mask, size=(h, w), mode='nearest')
        # get wide edges
        edges = self.get_wide_edges(label)
        # get weight mask
        weight = edges * self.opt.wide_edge + (1 - edges)
        return weight

    def loss(self, input, target_is_real, for_discriminator=True, label=None):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if not self.opt.remove_background:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(input - 1, self.get_zero_tensor(input))
                        if self.opt.wide_edge > 1.0:
                            minval = minval * self.get_weight_mask(input, label)
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-input - 1, self.get_zero_tensor(input))
                        if self.opt.wide_edge > 1.0:
                            minval = minval * self.get_weight_mask(input, label)
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(input)
            else:
                # resize the label
                n, c, h, w = input.size()
                label1 = F.interpolate(label, size=(h, w), mode='nearest')
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min((input - 1)*label1, self.get_zero_tensor(input))
                        loss = -torch.sum(minval) / (torch.sum(label1)*c+1e-5)
                    else:
                        minval = torch.min((-input - 1)*label1, self.get_zero_tensor(input))
                        loss = -torch.sum(minval) / (torch.sum(label1)*c+1e-5)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.sum(input*label1) / (torch.sum(label1)*c+1e-5)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True, label=None):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator, label.detach())
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator, label.detach())


# GAN feature loss
class GANFeatLoss(nn.MSELoss):
    def __init__(self, opt=None):
        super(GANFeatLoss, self).__init__()
        self.opt = opt
        self.criterionFeat = torch.nn.L1Loss()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def L1_loss_mask(self, input, target, label):
        # resize the label to math the size of feature
        n, c, h, w = input.size()
        label1 = F.interpolate(label, size=(h, w), mode='nearest')
        # cal L1 loss
        diff = torch.abs(input*label1 - target*label1)
        loss = diff.sum() / (label1.sum()*c + 1e-5)
        return loss

    def forward(self, pred_fake, pred_real, label=None):
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                if self.opt.remove_background:
                    unweighted_loss = self.L1_loss_mask(pred_fake[i][j], pred_real[i][j].detach(), label.detach())
                else:
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, opt=None):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.criterion_sum = nn.L1Loss(reduction='sum')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.opt = opt

    def L1_loss_mask(self, input, target, label=None):
        # resize the label to math the size of feature
        n, c, h, w = input.size()
        label1 = F.interpolate(label, size=(h, w), mode='nearest')
        # remove the background
        # cal L1 loss
        # diff = (input - target).abs()
        # loss = diff.sum() / (label.sum() + 1e-5)
        loss = self.criterion_sum(input*label1, target*label1) / (label1.sum()*c + 1e-5)
        return loss


    def forward(self, x, y, label=None):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if self.opt.remove_background:
                loss += self.weights[i] * self.L1_loss_mask(x_vgg[i], y_vgg[i].detach(), label.detach())
            else:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def gabor_fn(kernel_size, channel_in, channel_out, theta):
    # sigma_x = sigma
    # sigma_y = sigma.float() / gamma
    sigma_x = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()
    sigma_y = nn.Parameter(torch.ones(channel_out) * 3.0, requires_grad=False).cuda()
    Lambda = nn.Parameter(torch.ones(channel_out) * 4.0, requires_grad=False).cuda()
    psi = nn.Parameter(torch.ones(channel_out) * 0.0, requires_grad=False).cuda()

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb

def DoG_fn(kernel_size, channel_in, channel_out, theta):
    # params
    sigma_h = nn.Parameter(torch.ones(channel_out) * 1.0, requires_grad=False).cuda()
    sigma_l = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()
    sigma_y = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    gb = (torch.exp(-.5 * (x_theta ** 2 / sigma_h.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_h \
        - torch.exp(-.5 * (x_theta ** 2 / sigma_l.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_l) \
         / (1.0/sigma_h - 1.0/sigma_l)

    return gb

# L1 loss of orientation map
class L1OLoss(nn.Module):
    def __init__(self, opt, channel_in=1, channel_out=1, stride=1, padding=8):
        super(L1OLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.opt = opt
        self.mode = opt.orient_filter
        self.Tensor = torch.cuda.FloatTensor

        self.numKernels = 32
        self.kernel_size = 17
        # self.sigma_x = nn.Parameter(torch.ones(channel_out)*2.0, requires_grad=False).cuda()
        # self.sigma_y = nn.Parameter(torch.ones(channel_out)*3.0, requires_grad=False).cuda()
        # self.Lambda = nn.Parameter(torch.ones(channel_out)*4.0, requires_grad=False).cuda()
        # self.psi = nn.Parameter(torch.ones(channel_out)*0.0, requires_grad=False).cuda()

    def calOrientationGabor(self, image):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False).cuda()
            GaborKernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, theta)
            GaborKernel = GaborKernel.float()
            response = F.conv2d(image, GaborKernel, stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float()
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        confidenceTensor = (F.tanh(confidenceTensor)+1)/2.0 # [0, 1]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1)
        # cal the angle a
        orientTensor = maxResTensor * math.pi / self.numKernels
        orientTensor = torch.unsqueeze(orientTensor, 1)
        # cal the sin2a and cos2a
        orientTwoChannel = torch.cat([torch.sin(2*orientTensor), torch.cos(2*orientTensor)], dim=1) * confidenceTensor
        return orientTwoChannel, confidenceTensor

    def calOrientationDoG(self, image, mask):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False).cuda()
            DoGKernel = DoG_fn(self.kernel_size, self.channel_in, self.channel_out, theta)
            DoGKernel = DoGKernel.float()
            response = F.conv2d(image, DoGKernel, stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float()
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        # confidenceTensor = (F.tanh(confidenceTensor)+1)/2.0 # [0, 1]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1)
        confidenceTensor = confidenceTensor * mask
        confidenceTensor = confidenceTensor / torch.max(confidenceTensor)
        mask = confidenceTensor <= 0
        confidenceTensor = confidenceTensor * (1 - mask).float()
        # cal the angle a
        orientTensor = maxResTensor * math.pi / self.numKernels
        orientTensor = torch.unsqueeze(orientTensor, 1)
        # cal the sin2a and cos2a
        orientTwoChannel = torch.cat([torch.sin(2*orientTensor), torch.cos(2*orientTensor)], dim=1) * confidenceTensor
        return orientTwoChannel, confidenceTensor


    def forward(self, fake_image0, orientation_label0, input_semantics):
        # constraint the area of hair, input_semantics is one-hot map
        hair_mask = input_semantics[:,1,:,:]
        hair_mask = torch.unsqueeze(hair_mask, 1)
        # RGB to gray
        fake_image = (fake_image0+1)/2.0*255
        gray = 0.299*fake_image[:,0,:,:] + 0.587*fake_image[:,1,:,:] + 0.144*fake_image[:,2,:,:]
        gray = torch.unsqueeze(gray, 1)
        # cal orientation map with two channels
        if 'gabor' in self.mode:
            orientation_fake, confidence = self.calOrientationGabor(gray) # [n, 2, h, w]
        else:
            orientation_fake, confidence = self.calOrientationDoG(gray, hair_mask)  # [n, 2, h, w]
        # transfor the label from one channel to two channels
        if not self.opt.use_ig:
            orientation_label = orientation_label0 / 255 * math.pi
            orientation_mask = torch.cat([torch.sin(2*orientation_label), torch.cos(2*orientation_label)], dim=1)
        else:
            orientation_mask = orientation_label0
        # print(hair_mask.shape, orientation_fake.shape)
        orientation_fake = orientation_fake * hair_mask
        orientation_mask = orientation_mask * hair_mask
        # cal L1 loss and the log confidence loss
        orient_loss = self.criterion(orientation_fake, orientation_mask.detach())
        if 'gabor' in self.mode:
            confidence = torch.clamp(confidence, 0.001, 1)
            confidence_loss = -torch.sum(torch.log(confidence)*hair_mask)/torch.sum(hair_mask)
        else:
            confidence_gt = (hair_mask * 0 + 1) * hair_mask
            confidence_gt.requires_grad_(False)
            confidence = confidence*hair_mask
            confidence_loss = torch.sum(torch.abs(confidence-confidence_gt.detach())) / (torch.sum(hair_mask) + 1e-5)

        return orient_loss, confidence_loss

# cal l1 loss of the fake image and the real image outside the region of hair
class RGBBackgroundL1Loss(nn.Module):
    def __init__(self):
        super(RGBBackgroundL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, fake, input_semantics, image_tag):
        # constraint the area of hair, input_semantics is one-hot map
        background_mask = input_semantics[:,0,:,:]
        background_mask = torch.unsqueeze(background_mask, 1)

        fake_background = fake * background_mask
        image_no_hair = image_tag * background_mask
        return self.criterion(fake_background, image_no_hair.detach())

# cal l1 loss of the fake image and the real image in Lab color space
class LabColorLoss(nn.Module):
    def __init__(self, opt):
        super(LabColorLoss, self).__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor
        self.criterion = nn.L1Loss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    # cal lab written by tzt
    def func(self, x):
        mask = (x > 0.008856).float()
        return x ** (1 / 3) *mask + (7.787 * x + 0.137931) * (1 - mask)
    def RGB2Lab(self, input):
        # the range of input is from 0 to 1
        input_x = 0.412453 * input[:, 0, :, :] + 0.357580 * input[:, 1, :, :] + 0.180423 * input[:, 2, :, :]
        input_y = 0.212671 * input[:, 0, :, :] + 0.715160 * input[:, 1, :, :] + 0.072169 * input[:, 2, :, :]
        input_z = 0.019334 * input[:, 0, :, :] + 0.119193 * input[:, 1, :, :] + 0.950227 * input[:, 2, :, :]
        # normalize
        # input_xyz = input_xyz / 255.0
        input_x = input_x / 0.950456 # X
        input_y = input_y / 1.0 # Y
        input_z = input_z / 1.088754 # Z

        fx = self.func(input_x)
        fy = self.func(input_y)
        fz = self.func(input_z)

        Y_mask = (input_y > 0.008856).float()
        input_l = (116.0 * fy - 16.0) * Y_mask + 903.3 * input_y * (1 - Y_mask) # L
        input_a = 500 * (fx - fy) # a
        input_b = 200 * (fy - fz) # b

        input_l = torch.unsqueeze(input_l, 1)
        input_a = torch.unsqueeze(input_a, 1)
        input_b = torch.unsqueeze(input_b, 1)
        return torch.cat([input_l, input_a, input_b],1)
    # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[1 - mask] = 7.787 * input[1 - mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        """Change RGB color to XYZ color

        Args:
            input: 4-D tensor, [B, C, H, W]
        """
        assert input.size(1) == 3

        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC

        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW

        # output = output / 255.0

        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1

        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3

        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][1 - mask] = 903.3 * input[:, 1, :, :][1 - mask]

        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])

        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])

        return output

    def cal_weight(self, tag_image, mask):
        n,c,h,w = tag_image.size()
        raw = np.load(self.opt.weight_dir)
        weight = self.FloatTensor(raw)
        weight = torch.unsqueeze(torch.unsqueeze(weight, 0), 0)
        weight = weight.repeat(n, 1, 1, 1)
        weight[weight == 0] = 1
        weight = weight.max() / weight
        weight[weight > self.opt.Lab_weight_th] = self.opt.Lab_weight_th

        image_a = torch.unsqueeze(tag_image[:,1,:,:], 1)
        image_b = torch.unsqueeze(tag_image[:,2,:,:], 1)
        m = torch.cat([image_b, image_a], 1) + 128
        m[m < 0] = 0
        m[m > 255] = 255
        m = m.int().float()
        m = (m - 127.5) / 127.5
        m = m.permute([0, 2, 3, 1])

        weight_mask = grid_sample(weight, m, mode='nearest')
        weight_mask = weight_mask * mask
        weight_mask[weight_mask == 0] = 1

        return weight_mask

    def forward(self, fake, real, mask=None):
        # normalize to 0~1
        fake_RGB = (fake + 1) / 2.0
        real_RGB = (real + 1) / 2.0
        ## from RGB to Lab by tzt
        # fake_Lab = self.RGB2Lab(fake_RGB)
        # real_Lab = self.RGB2Lab(real_RGB)
        # from RGB to Lab by liuqk
        fake_xyz = self.rgb2xyz(fake_RGB)
        fake_Lab = self.xyz2lab(fake_xyz)
        real_xyz = self.rgb2xyz(real_RGB)
        real_Lab = self.xyz2lab(real_xyz)
        # cal loss
        if self.opt.balance_Lab:
            weight_mask = self.cal_weight(real_Lab, mask)
            diff = torch.abs(fake_Lab[:,1:,:,:] - real_Lab[:,1:,:,:].detach())
            w_diff = weight_mask * diff
            lab_loss = torch.mean(w_diff)
        else:
            lab_loss = self.criterion(fake_Lab[:,1:,:,:], real_Lab[:,1:,:,:].detach())
        # if (lab_loss != lab_loss).sum() > 0:
        #     pdb.set_trace()
        return lab_loss

# cal avarage Lab loss in hair
class HairAvgLabLoss(nn.Module):
    def __init__(self, opt):
        super(HairAvgLabLoss, self).__init__()
        self.opt = opt
        self.criterion = nn.L1Loss()
        self.FloatTensor = torch.cuda.FloatTensor
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[1 - mask] = 7.787 * input[1 - mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][1 - mask] = 903.3 * input[:, 1, :, :][1 - mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output
    def cal_hair_avg(self, input, mask):
        x = input * mask
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        return avg
    def cal_weight(self, tag_image):
        n, c, h, w = tag_image.size()
        raw = np.load(self.opt.weight_dir)
        weight = self.FloatTensor(raw)
        weight = torch.unsqueeze(torch.unsqueeze(weight, 0), 0)
        weight = weight.repeat(n, 1, 1, 1)
        weight[weight == 0] = 1
        weight = weight.max() / weight
        weight[weight > self.opt.Lab_weight_th] = self.opt.Lab_weight_th

        image_a = torch.unsqueeze(tag_image[:,1,:,:], 1)
        image_b = torch.unsqueeze(tag_image[:,2,:,:], 1)
        m = torch.cat([image_b, image_a], 1) + 128
        m[m < 0] = 0
        m[m > 255] = 255
        # m = min(max(m, -128), 127) + 128 # range from 0 to 255
        m = m.int().float()
        m = (m - 127.5) / 127.5
        m = m.permute([0, 2, 3, 1])

        weight_mask = grid_sample(weight, m, mode='nearest')

        return weight_mask
    def forward(self, fake, real, mask_fake, mask_real):
        # the mask is [n,1,h,w]
        # normalize to 0~1
        fake_RGB = (fake + 1) / 2.0
        real_RGB = (real + 1) / 2.0
        # from RGB to Lab by liuqk
        fake_xyz = self.rgb2xyz(fake_RGB)
        fake_Lab = self.xyz2lab(fake_xyz)
        real_xyz = self.rgb2xyz(real_RGB)
        real_Lab = self.xyz2lab(real_xyz)
        # cal average value
        fake_Lab_avg = self.cal_hair_avg(fake_Lab, mask_fake)
        real_Lab_avg = self.cal_hair_avg(real_Lab, mask_real)
        if self.opt.balance_Lab:
            weight_mask = self.cal_weight(real_Lab_avg)
            diff = torch.abs(fake_Lab_avg[:,1:,:,:] - real_Lab_avg[:,1:,:,:].detach())
            w_diff = weight_mask * diff
            loss = torch.mean(w_diff)
        else:
            loss = self.criterion(fake_Lab_avg[:,1:,:,:], real_Lab_avg[:,1:,:,:].detach())
        return loss

# calculate the style loss and content loss
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std_mask(feat, mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    mask1 = mask.view(N, 1, -1)
    feat1 = feat.view(N, C, -1) * mask1
    # cal the mean
    feat_mean = feat1.sum(dim=2) / (mask1.sum(dim=2) + eps)
    feat_mean = feat_mean.view(N, C, 1)
    # cal the std
    feat_var = ((feat1 - feat_mean) * mask1)**2
    feat_var = (feat_var.sum(dim=2) / (mask1.sum(dim=2) + eps)) + eps
    feat_std = feat_var.sqrt()
    # feat_std_isNan = torch.isnan(feat_std).sum()
    # if feat_std_isNan > 0:
    #     print('feat std contains nan')
    # reshape the mean and std
    feat_mean = feat_mean.view(N, C, 1, 1)
    feat_std = feat_std.view(N, C, 1, 1)
    return feat_mean, feat_std

class StyleContentLoss(nn.Module):
    def __init__(self, opt=None):
        super(StyleContentLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_sum = nn.MSELoss(reduction='sum')
        self.opt = opt

    def calc_content_loss(self, input, target, content_label=None):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        if not self.opt.remove_background:
            return self.mse_loss(input, target)
        else:
            # resize the label
            n, c, h, w = input.size()
            label = F.interpolate(content_label, size=(h, w), mode='nearest')
            # remove the background of features
            ret = (input * label - target * label) ** 2
            ret = torch.sum(ret)
            # cal the loss
            return ret / (torch.sum(label)*c + 1e-5)

    def calc_style_loss(self, input, target, style_label=None, content_label=None):
        # input is the style feature, target is the fake feature
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        if not self.opt.remove_background:
            input_mean, input_std = calc_mean_std(input)
            target_mean, target_std = calc_mean_std(target)
            return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)
        else:
            # resize the label
            n, c, h, w = input.size()
            content_label1 = F.interpolate(content_label, size=(h, w), mode='nearest')
            style_label1 = F.interpolate(style_label, size=(h, w), mode='nearest')
            input_mean, input_std = calc_mean_std_mask(input, style_label1)
            target_mean, target_std = calc_mean_std_mask(target, content_label1)
            return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)


    def forward(self, fake_image, style_image, content_image, style_label=None, content_label=None):
        # style_image is the reference image, content_image is the target image
        fake_vgg, style_vgg, content_vgg = self.vgg(fake_image), self.vgg(style_image), self.vgg(content_image)
        if self.opt.remove_background:
            loss_c = self.calc_content_loss(fake_vgg[-1], content_vgg[-1].detach(), content_label.detach())
            loss_s = 0
            for i in range(len(fake_vgg)):
                loss_s += self.calc_style_loss(fake_vgg[i], style_vgg[i].detach(), style_label.detach(), content_label.detach())
            return loss_c, loss_s
        else:
            loss_c = self.calc_content_loss(fake_vgg[-1], content_vgg[-1].detach())
            loss_s = 0
            for i in range(len(fake_vgg)):
                loss_s += self.calc_style_loss(fake_vgg[i], style_vgg[i].detach())
            return loss_c, loss_s


