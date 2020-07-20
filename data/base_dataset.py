"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

def pad_zeros(input, pad_th):
    '''
    :param input: type: PIL Image
    :param pad_th: int
    :return:
    '''
    img = np.array(input)
    size = img.shape
    if len(size) == 2:
        H, W = size[0], size[1]
        pad_img = np.zeros((H + pad_th, W + pad_th))
        pad_img[int(pad_th / 2):int(pad_th / 2) + H, int(pad_th / 2):int(pad_th / 2) + W] = img
    else:
        H, W, C = size[0], size[1], size[2]
        pad_img = np.zeros((H+pad_th, W+pad_th, C))
        pad_img[int(pad_th/2):int(pad_th/2)+H, int(pad_th/2):int(pad_th/2)+W, :] = img
    pad_img = np.uint8(pad_img)
    # plt.imshow(pad_img)
    # plt.show()
    return Image.fromarray(pad_img)

def single_inference_dataLoad(opt):
    base_dir = opt.data_dir
    subset = opt.subset
    label_ref_dir = base_dir + '/' + subset + '_labels/' + opt.inference_ref_name + '.png'
    label_tag_dir = base_dir + '/' + subset + '_labels/' + opt.inference_tag_name + '.png'
    orient_tag_dir = base_dir + '/' + subset + '_dense_orients/' + opt.inference_tag_name + '_orient_dense.png'
    orient_ref_dir = base_dir + '/' + subset + '_dense_orients/' + opt.inference_orient_name + '_orient_dense.png'
    orient_mask_dir = base_dir + '/' + subset + '_labels/' + opt.inference_orient_name + '.png'
    image_ref_dir = base_dir + '/' + subset + '_images/' + opt.inference_ref_name + '.jpg'
    image_tag_dir = base_dir + '/' + subset + '_images/' + opt.inference_tag_name + '.jpg'

    label_ref = Image.open(label_ref_dir)
    label_tag = Image.open(label_tag_dir)
    orient_mask = Image.open(orient_mask_dir)
    orient_tag = Image.open(orient_tag_dir)
    orient_ref = Image.open(orient_ref_dir)
    image_ref = Image.open(image_ref_dir)
    image_tag = Image.open(image_tag_dir)

    # add zeros
    if opt.add_zeros:
        label_ref = pad_zeros(label_ref, opt.add_th)
        label_tag = pad_zeros(label_tag, opt.add_th)
        orient_mask = pad_zeros(orient_mask, opt.add_th)
        orient_tag = pad_zeros(orient_tag, opt.add_th)
        orient_ref = pad_zeros(orient_ref, opt.add_th)
        image_ref = pad_zeros(image_ref, opt.add_th)
        image_tag = pad_zeros(image_tag, opt.add_th)

    # orient, label = RandomErasure(orient, label)
    # label process
    params = get_params(opt, label_ref.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_ref_tensor = transform_label(label_ref) * 255.0
    label_ref_tensor[label_ref_tensor == 255] = opt.label_nc
    label_ref_tensor = torch.unsqueeze(label_ref_tensor, 0)

    if opt.expand_tag_mask:
        label_tag_array = np.array(label_tag)
        di_k = 25
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))
        label_tag_array = cv2.dilate(label_tag_array, dilate_kernel)
        label_tag = Image.fromarray(np.uint8(label_tag_array)).convert('L')

    label_tag_tensor = transform_label(label_tag) * 255.0
    label_tag_tensor[label_tag_tensor == 255] = opt.label_nc
    label_tag_tensor = torch.unsqueeze(label_tag_tensor, 0)

    orient_mask_tensor = transform_label(orient_mask) * 255.0
    orient_mask_tensor[orient_mask_tensor == 255] = opt.label_nc
    orient_mask_tensor = torch.unsqueeze(orient_mask_tensor, 0)
    # if opt.expand_tag_mask:
    #     k = opt.expand_value
    #     p = int(k / 2)
    #     orient_mask_tensor = F.max_pool2d(orient_mask_tensor, kernel_size=k, stride=1, padding=p)

    # rgb orientation maps
    if opt.use_ig and not opt.no_orientation:
        orient_tag_rgb = trans_orient_to_rgb(np.array(orient_ref), np.array(label_tag), np.array(orient_mask))
        orient_rgb_tensor = transform_label(orient_tag_rgb)
        orient_rgb_tensor = torch.unsqueeze(orient_rgb_tensor, 0)
        orient_rgb_tensor = orient_rgb_tensor * label_tag_tensor
    else:
        orient_rgb_tensor = torch.tensor(0)

    # hole mask
    if opt.use_ig:
        if opt.inference_orient_name == opt.inference_tag_name:
            hole = np.array(label_tag)
            hole = generate_hole(hole, np.array(orient_mask))
            hole_tensor = transform_label(hole) * 255.0
            hole_tensor = torch.unsqueeze(hole_tensor, 0)
        else:
            hole_tensor = label_tag_tensor - orient_mask_tensor * label_tag_tensor

    else:
        hole_tensor = torch.tensor(0)


    # generate noise
    noise = generate_noise(opt.crop_size, opt.crop_size)
    noise_tensor = torch.tensor(noise).permute(2, 0, 1)
    noise_tensor = torch.unsqueeze(noise_tensor, 0)

    image_ref = image_ref.convert('RGB')
    if opt.color_jitter:
        transform_image = get_transform(opt, params, color=True)
    else:
        transform_image = get_transform(opt, params)
    image_ref_tensor = transform_image(image_ref)
    image_ref_tensor = torch.unsqueeze(image_ref_tensor, 0)

    image_tag = image_tag.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tag_tensor = transform_image(image_tag)
    image_tag_tensor = torch.unsqueeze(image_tag_tensor, 0)

    orient_tensor = transform_label(orient_tag) * 255
    orient_tensor = torch.unsqueeze(orient_tensor, 0)

    data = {'label_ref': label_ref_tensor,
            'label_tag': label_tag_tensor,
            'instance': torch.tensor(0),
            'image_ref': image_ref_tensor,
            'image_tag': image_tag_tensor,
            'path': image_tag_dir,
            'orient': orient_tensor,
            'hole': hole_tensor,
            'orient_rgb': orient_rgb_tensor,
            'noise': noise_tensor
            }
    return data

def demo_inference_dataLoad(opt, ref_label_dir, tag_label, mask_orient, ref_orient, ref_image, tag_image, orient_stroke=None, mask_stroke=None, mask_hole=None):
    '''
    :param opt:
    :param ref_label_dir:
    :param tag_label:
    :param mask_orient:
    :param ref_orient:
    :param ref_image:
    :param tag_image:
    :param orient_stroke: type: np.array, shape: 512*512*3, range: [0, 255]
    :param mask_stroke: type: np.array, range: {0, 1}, shape: 512*512
    :param mask_hole: type: np.array, range: {0, 1}, shape: 512*512
    :return:
    '''
    label_ref = Image.open(ref_label_dir)
    label_tag = Image.fromarray(np.uint8(tag_label))
    orient_mask = Image.fromarray(np.uint8(mask_orient))
    orient_ref = Image.fromarray(np.uint8(ref_orient))
    image_ref = ref_image
    image_tag = tag_image

    # orient, label = RandomErasure(orient, label)
    # label process
    params = get_params(opt, label_ref.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_ref_tensor = transform_label(label_ref) * 255.0
    label_ref_tensor[label_ref_tensor == 255] = opt.label_nc
    label_ref_tensor = torch.unsqueeze(label_ref_tensor, 0)

    if opt.expand_tag_mask:
        label_tag_array = np.array(label_tag)
        di_k = 25
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))
        label_tag_array = cv2.dilate(label_tag_array, dilate_kernel)
        label_tag = Image.fromarray(np.uint8(label_tag_array)).convert('L')

    label_tag_tensor = transform_label(label_tag) * 255.0
    label_tag_tensor[label_tag_tensor == 255] = opt.label_nc
    label_tag_tensor = torch.unsqueeze(label_tag_tensor, 0)

    orient_mask_tensor = transform_label(orient_mask) * 255.0
    orient_mask_tensor[orient_mask_tensor == 255] = opt.label_nc
    orient_mask_tensor = torch.unsqueeze(orient_mask_tensor, 0)

    # rgb orientation maps
    orient_tag_rgb = trans_orient_to_rgb(np.array(orient_ref), np.array(label_tag), np.array(orient_mask))
    orient_rgb_tensor = transform_label(orient_tag_rgb)
    orient_rgb_tensor = torch.unsqueeze(orient_rgb_tensor, 0)
    orient_rgb_tensor = orient_rgb_tensor * label_tag_tensor
    orient_rgb_mask = orient_mask_tensor * label_tag_tensor


    # hole mask
    if mask_hole is None:
        hole_tensor = label_tag_tensor - orient_mask_tensor * label_tag_tensor
    else:
        mask_hole_img = Image.fromarray(np.uint8(mask_hole))
        hole_tensor = transform_label(mask_hole_img) * 255.0
        hole_tensor = torch.unsqueeze(hole_tensor, 0) * label_tag_tensor

    # orient_stroke
    if orient_stroke is not None:
        orient_stroke_img = Image.fromarray(np.uint8(orient_stroke))
        # orient_stroke_img.save('./inference_samples/orient_stroke.png')
        orient_stroke_tensor = transform_label(orient_stroke_img)
        orient_stroke_tensor = torch.unsqueeze(orient_stroke_tensor, 0)
        orient_stroke_tensor = orient_stroke_tensor * label_tag_tensor
    else:
        orient_stroke_tensor = torch.tensor(0)

    # mask_stroke
    if mask_stroke is not None:
        mask_stroke_img = Image.fromarray(np.uint8(mask_stroke))
        mask_stroke_tensor = transform_label(mask_stroke_img) * 255.0
        mask_stroke_tensor = torch.unsqueeze(mask_stroke_tensor, 0) * label_tag_tensor
    else:
        mask_stroke_tensor = torch.tensor(0)


    # generate noise
    noise = generate_noise(opt.crop_size, opt.crop_size)
    noise_tensor = torch.tensor(noise).permute(2, 0, 1)
    noise_tensor = torch.unsqueeze(noise_tensor, 0)

    image_ref = image_ref.convert('RGB')
    if opt.color_jitter:
        transform_image = get_transform(opt, params, color=True)
    else:
        transform_image = get_transform(opt, params)
    image_ref_tensor = transform_image(image_ref)
    image_ref_tensor = torch.unsqueeze(image_ref_tensor, 0)

    image_tag = image_tag.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tag_tensor = transform_image(image_tag)
    image_tag_tensor = torch.unsqueeze(image_tag_tensor, 0)
    #
    orient_tensor = transform_label(orient_ref) * 255
    orient_tensor = torch.unsqueeze(orient_tensor, 0)

    data = {'label_ref': label_ref_tensor,
            'label_tag': label_tag_tensor,
            'instance': torch.tensor(0),
            'image_ref': image_ref_tensor,
            'image_tag': image_tag_tensor,
            'path': None,
            'orient': orient_tensor,
            'hole': hole_tensor,
            'orient_rgb': orient_rgb_tensor,
            'orient_rgb_mask': orient_rgb_mask,
            'noise': noise_tensor,
            'orient_stroke': orient_stroke_tensor,
            'mask_stroke': mask_stroke_tensor
            }
    return data

def show_training_data(data):
    noise = data['noise']
    orient_rgb = data['orient_rgb']
    hole = data['hole']
    image_ref = data['image_ref']
    image_tag = data['image_tag']
    # trans , noise and orient_rgb is range from 0 to 1
    noise = noise.permute(1,2,0).numpy()
    orient_rgb = orient_rgb.permute(1,2,0).numpy()
    hole = hole.permute(1,2,0).numpy()
    image_ref = (image_ref.permute(1,2,0).numpy() + 1) / 2
    image_tag = (image_tag.permute(1, 2, 0).numpy() + 1) / 2
    orient_noise = orient_rgb * (1 - hole) + noise * hole
    # plt
    plt.subplot(2,2,1)
    plt.imshow(orient_rgb)
    plt.subplot(2,2,2)
    plt.imshow(orient_noise)
    plt.subplot(2,2,3)
    plt.imshow(image_ref)
    plt.subplot(2,2,4)
    plt.imshow(image_tag)
    plt.show()


def RandomErasure(orient, label):
    import math
    orient_array = np.array(orient)
    H, W = orient_array.shape
    if abs(orient_array).max() == 0:
        return orient, label
    else:
        coord = np.where(orient_array != 0)
        nums = len(coord[0])
        th = random.uniform(0.3, 1.5)
        crop_nums = int(th * nums)
        rr = int(crop_nums / math.pi)

        center_idx = random.randint(0, nums-1)

        center_h = coord[0][center_idx]
        center_w = coord[1][center_idx]

        tmp_h = np.array(range(H))
        tmp_h = tmp_h.repeat(W).reshape(H, W)
        tmp_w = np.array(range(W))
        tmp_w = np.tile(tmp_w, H).reshape(H, W)

        mask = ((tmp_h - center_h) ** 2 + (tmp_w - center_w) ** 2) < rr
        mask = mask.astype(np.float)

        orient_array = orient_array * (1-mask)
        orient_array = Image.fromarray(np.uint8(orient_array))
        label_array = np.array(label) * (1-mask)
        label_array = Image.fromarray(np.uint8(label_array))
        return orient_array, label_array

def generate_hole(mask, orient_mask):
    import math
    H, W = orient_mask.shape
    if abs(orient_mask).max() == 0:
        return Image.fromarray(np.uint8(orient_mask)).convert('L')
    else:
        coord = np.where(orient_mask != 0)
        nums = len(coord[0])
        th = random.uniform(0.5, 1.2)
        crop_nums = int(th * nums)
        rr = int(crop_nums / math.pi)

        center_idx = random.randint(0, nums-1)

        center_h = coord[0][center_idx]
        center_w = coord[1][center_idx]

        tmp_h = np.array(range(H))
        tmp_h = tmp_h.repeat(W).reshape(H, W)
        tmp_w = np.array(range(W))
        tmp_w = np.tile(tmp_w, H).reshape(H, W)

        tmp_mask = ((tmp_h - center_h) ** 2 + (tmp_w - center_w) ** 2) < rr
        tmp_mask = tmp_mask.astype(np.float)
        hole_mask = orient_mask * tmp_mask + (mask - orient_mask)
        hole = Image.fromarray(np.uint8(hole_mask)).convert('L')
        return hole

def trans_orient_to_rgb(orient, label, orient_label=None):
    import math
    # orient is the dense orient map which ranges from 0 to 255, orient_label is the mask which matches the orient
    # if orient_label is None, that means label matches the orient
    orient_mask = orient / 255.0 * math.pi
    H, W = orient_mask.shape
    orient_rgb = np.zeros((H, W, 3))
    orient_rgb[..., 1] = (np.sin(2 * orient_mask)+1)/2
    orient_rgb[..., 0] = (np.cos(2 * orient_mask)+1)/2
    orient_rgb[...,2] = 0.5

    if orient_label is None:
        orient_rgb *= label[...,np.newaxis]
        orient_rgb = orient_rgb * 255.0
        orient_rgb = Image.fromarray(np.uint8(orient_rgb)).convert('RGB')
        # orient_rgb.save('./inference_samples/orient_before_trans.png')
        return orient_rgb
    else:
        orient_rgb *= orient_label[..., np.newaxis]
        orient_rgb = orient_rgb * 255.0
        orient_rgb = Image.fromarray(np.uint8(orient_rgb)).convert('RGB')
        # orient_rgb.save('./inference_samples/orient_before_trans.png')
        return orient_rgb

def generate_noise(width, height):
    weight = 1.0
    weightSum = 0.0
    noise = np.zeros((height, width, 3)).astype(np.float32)
    while width >= 8 and height >= 8:
        noise += cv2.resize(np.random.normal(loc = 0.5, scale = 0.25, size = (int(height), int(width), 3)), dsize = (noise.shape[0], noise.shape[1])) * weight
        weightSum += weight
        width //= 2
        height //= 2
    return noise / weightSum

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, color=False):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if color:
        transform_list += [transforms.ColorJitter(brightness=0.1, contrast=0.01, saturation=0.01, hue=0.01)]

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
