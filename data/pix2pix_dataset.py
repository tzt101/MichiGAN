"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from data.base_dataset import BaseDataset, get_params, get_transform, generate_hole, trans_orient_to_rgb, generate_noise, show_training_data
from PIL import Image
import util.util as util
import os
import numpy as np
import torch
import random



class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt, step=1):
        self.opt = opt
        self.step = step

        label_paths, image_paths, instance_paths, orient_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)
        if not opt.no_orientation:
            util.natural_sort(orient_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]
        orient_paths = orient_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        self.orient_paths = orient_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # tag Label
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # reference Label
        if self.step == 1:
            index_ref = index
        else:
            index_ref = random.randint(0, len(self.label_paths)-1)
        label_path_ref = self.label_paths[index_ref]
        label_ref = Image.open(label_path_ref)
        label_tensor_ref = transform_label(label_ref) * 255.0
        label_tensor_ref[label_tensor_ref == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input tag image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # input reference image (style images)
        image_path_ref = self.image_paths[index_ref]
        image_ref = Image.open(image_path_ref).convert('RGB')
        if self.opt.color_jitter:
            transform_image = get_transform(self.opt, params, color=True)
        image_tensor_ref = transform_image(image_ref)


        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        # if using orientation maps
        if self.opt.no_orientation:
            orient_tensor = 0
        else:
            orient_path = self.orient_paths[index]
            orient = Image.open(orient_path)
            orient_tensor = transform_label(orient)*255

        # rgb orientation maps
        index_orient_ref = random.randint(0, len(self.label_paths) - 1)
        orient_rgb = Image.open(self.orient_paths[index_orient_ref])
        orient_mask = Image.open(self.label_paths[index_orient_ref])
        orient_random_param = random.random()
        orient_random_th = 2
        if self.opt.use_ig and not self.opt.no_orientation:
            transform_orient_rgb = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            if orient_random_param < orient_random_th:
                # use the target orient with erasure
                orient_rgb = trans_orient_to_rgb(np.array(orient), np.array(label))
                orient_rgb_tensor = transform_orient_rgb(orient_rgb) * label_tensor
            else:
                # use the reference orient that not match the reference image, this is the other random orient
                # print('index of sample', index, index_ref, index_orient_ref)
                orient_rgb = trans_orient_to_rgb(np.array(orient_rgb), np.array(label), np.array(orient_mask))
                orient_rgb_tensor = transform_orient_rgb(orient_rgb)
                orient_rgb_tensor = orient_rgb_tensor * label_tensor
        else:
            orient_rgb_tensor = torch.tensor(0)

        # process orient mask
        orient_mask_tensor = transform_label(orient_mask) * 255.0
        orient_mask_tensor[orient_mask_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # hole mask
        if self.opt.use_ig:
            if orient_random_param < orient_random_th:
                hole = np.array(label)
                hole = generate_hole(hole, np.array(orient_mask))
                hole_tensor = transform_label(hole) * 255.0
            else:
                hole_tensor = label_tensor - orient_mask_tensor * label_tensor
        else:
            hole_tensor = 0

        # generate noise
        noise = generate_noise(self.opt.crop_size, self.opt.crop_size)
        noise_tensor = torch.tensor(noise).permute(2, 0, 1)

        # # random: the reference label and image are the same with reference or not
        # if self.opt.only_tag:
        #     if not self.opt.use_blender:
        #         image_tensor_ref = image_tensor.clone()
        #         label_tensor_ref = label_tensor.clone()
        #     else:
        #         if random.random() < 2:
        #             image_tensor_ref = image_tensor.clone()
        #             label_tensor_ref = label_tensor.clone()
        # else:
        #     if random.random() < 0.2:
        #         image_tensor_ref = image_tensor.clone()
        #         label_tensor_ref = label_tensor.clone()


        input_dict = {'label_tag': label_tensor,
                      'label_ref': label_tensor_ref,
                      'instance': instance_tensor,
                      'image_tag': image_tensor,
                      'image_ref': image_tensor_ref,
                      'path': image_path_ref,
                      'orient': orient_tensor,
                      'hole': hole_tensor,
                      'orient_rgb': orient_rgb_tensor,
                      'noise': noise_tensor
                      }
        # # show and debug
        # show_training_data(input_dict)
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
