"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import random
import os


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        # parser.set_defaults(load_size=load_size)
        # parser.set_defaults(crop_size=256)
        # parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=2)
        parser.set_defaults(contain_dontcare_label=False)

        # parser.add_argument('--data_dir', type=str, default='/mnt/lvdisk1/tzt/HairSynthesis/SPADE-master/datasets/FFHQ',
        #                     help='path to the directory that contains training & val data')
        parser.add_argument('--label_dir', type=str, default='train_labels',
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='train_images',
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        parser.add_argument('--orient_dir', type=str, default='train_dense_orients',
                            help='path to the directory that contains orientation mask')
        parser.add_argument('--clear', type=str, default='',
                            help='[ |clear_], clear_ means use the selected training data')

        return parser

    def get_paths(self, opt):

        # combine data_dir and others
        label_dir = os.path.join(opt.data_dir, opt.clear+opt.label_dir)
        image_dir = os.path.join(opt.data_dir, opt.clear+opt.image_dir)
        orient_dir = os.path.join(opt.data_dir, opt.clear+opt.orient_dir)

        # label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        # image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        if len(opt.orient_dir) > 0:
            # orient_dir = opt.orient_dir
            orient_paths = make_dataset(orient_dir, recursive=False, read_cache=True)
        else:
            orient_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths, orient_paths
