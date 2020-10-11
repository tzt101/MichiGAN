import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_path', type=str, default='56000.jpg', help='Path to image')
parser.add_argument('--hairmask_path',type=str, default='56000.png', help='Path to hair mask')
parser.add_argument('--orientation_root', type=str, default='./', help='Root to save hair orientation map')

def DoG_fn(kernel_size, channel_in, channel_out, theta):
    # params
    sigma_h = nn.Parameter(torch.ones(channel_out) * 1.0, requires_grad=False)
    sigma_l = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False)
    sigma_y = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False)

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
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
    def __init__(self, channel_in=1, channel_out=1, stride=1, padding=8):
        super(orient, self).__init__()
        self.criterion = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.filter = DoG_fn

        self.numKernels = 32
        self.kernel_size = 17

    def calOrientation(self, image, mask=None):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False)
            filterKernel = self.filter(self.kernel_size, self.channel_in, self.channel_out, theta)
            filterKernel = filterKernel.float()
            response = F.conv2d(image, filterKernel, stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float() # range from 0 to 31
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1) 

        return maxResTensor, confidenceTensor

if __name__ == '__main__':
    args = parser.parse_args()
    # mkdir orientation root
    if not os.path.exists(args.orientation_root):
        os.mkdir(args.orientation_root)
        
    # Get structure
    image = Image.open(args.image_path)
    mask = np.array(Image.open(args.hairmask_path))
    if np.max(mask) > 1:
        mask = (mask > 130) * 1
    trans_image = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_tensor = trans_image(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    cal_orient = orient()
    fake_image = (image_tensor + 1) / 2.0 * 255
    gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
    gray = torch.unsqueeze(gray, 1)
    orient_tensor, confidence_tensor = cal_orient.calOrientation(gray)
    orient_tensor = orient_tensor * math.pi / 31 * 2
    mask_tensor = torch.from_numpy(mask).float()
    flow_x = torch.cos(orient_tensor) * confidence_tensor * mask_tensor
    flow_y = torch.sin(orient_tensor) * confidence_tensor * mask_tensor
    flow_x = torch.from_numpy(cv2.GaussianBlur(flow_x.numpy().squeeze(), (0, 0), 4))
    flow_y = torch.from_numpy(cv2.GaussianBlur(flow_y.numpy().squeeze(), (0, 0), 4))
    orient_tensor = torch.atan2(flow_y, flow_x) * 0.5
    orient_tensor[orient_tensor < 0] += math.pi
    orient_np = orient_tensor.numpy().squeeze() * 255. / math.pi * mask
    orient_save = Image.fromarray(np.uint8(orient_np))
    orient_save.save(os.path.join(args.orientation_root, args.image_path.split('/')[-1][:-4]+'.png'))
    # cv2.imwrite(args.orientation_root, orient_tensor.numpy().squeeze() * 255. / math.pi)
