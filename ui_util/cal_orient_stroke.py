import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import PIL
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt

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
class orient(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, stride=1, padding=8, mode='dog'):
        super(orient, self).__init__()
        self.criterion = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.filter = gabor_fn if mode == 'gabor' else DoG_fn

        self.numKernels = 32
        self.kernel_size = 17

    def calOrientation(self, image, mask=None):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out) * (math.pi * iOrient / self.numKernels),
                                 requires_grad=False).cuda()
            filterKernel = self.filter(self.kernel_size, self.channel_in, self.channel_out, theta)
            filterKernel = filterKernel.float()
            response = F.conv2d(image, filterKernel, stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float()
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        # confidenceTensor = (torch.tanh(confidenceTensor)+1)/2.0 # [0, 1]
        # confidenceTensor = confidenceTensor / torch.max(confidenceTensor)
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1)
        # print(torch.unique(confidenceTensor))
        # th = 0.4
        #
        # confidenceTensor[confidenceTensor >= th] = 1
        # confidenceTensor[confidenceTensor < th] = 0
        # print(torch.unique(confidenceTensor))
        # print(torch.sum(confidenceTensor))
        # confidenceTensor = torch.unsqueeze(confidenceTensor, 1) / torch.max(confidenceTensor)

        # cal the angle a
        orientTensor = maxResTensor * math.pi / self.numKernels
        orientTensor = torch.unsqueeze(orientTensor, 1)
        # cal the sin2a and cos2a
        orientTwoChannel = torch.cat([torch.sin(2 * orientTensor), torch.cos(2 * orientTensor)], dim=1)
        return orientTwoChannel, confidenceTensor

    def convert_orient_to_RGB_test(self, input, label):
        import torch
        label = label.float()
        input = input * label
        out_r = torch.unsqueeze(input[1, :, :] * label[0, ...] + (1 - label[0, ...]) * -1, 0)
        out_g = torch.unsqueeze(input[0, :, :] * label[0, ...] + (1 - label[0, ...]) * -1, 0)
        out_b = torch.unsqueeze(input[0, :, :] * 0 * label[0, ...] + (1 - label[0, ...]) * -1, 0)
        # print(out_b.shape)
        return torch.cat([out_r, out_g, out_b], dim=0)

    def stroke_to_orient(self, stroke_mask):
        '''
        :param stroke_mask: type: np.array, shape: 512*512, range: {0, 1}
        :return: type: np.array, shape: 512*512, range: [0,255]
        '''
        stroke_mask_img = Image.fromarray(np.uint8(stroke_mask*255))
        trans_label = transforms.Compose([transforms.ToTensor()])

        stroke_mask_tensor = trans_label(stroke_mask_img)

        stroke_mask_tensor = torch.unsqueeze(stroke_mask_tensor, 0).cuda()

        orient_tensor, confidence_tensor = self.calOrientation(stroke_mask_tensor)
        orient_tensor = orient_tensor * stroke_mask_tensor
        # vis
        orient_rgb = self.convert_orient_to_RGB_test(orient_tensor[0, ...], stroke_mask_tensor[0, ...])  # [3, h, w]
        orient_numpy = (np.transpose(orient_rgb.cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255.0

        return orient_numpy