"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='/mnt/lvdisk1/tzt/HairSynthesis/SPADE-master/results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='13', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=5000, help='how many test images to run')

        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1, gpu_ids='1')
        parser.set_defaults(gpu_ids='2')

        parser.add_argument('--source_dir', type=str, default='/mnt/lvdisk1/tzt/HairSynthesis/SPADE-master/results/SPADEBEncodeInpaint5B/')
        parser.add_argument('--source_file', type=str, default='comparison')
        parser.add_argument('--four_image_show', action='store_true', help='if specified, save the images contain the ref/tag/ori image.')
        parser.add_argument('--which_settings', type=str, default='spadeb512', help='which settings to test.')
        parser.add_argument('--which_random', type=str, default='orient', help='random the one of the input.')
        parser.add_argument('--input_relation', type=str, default='ref=tag!=ori', help='the relationship of three input frames.')
        parser.add_argument('--val_list_dir', type=str, default='data/val_image_list.txt', help='the text file which contains the image names to val.')

        parser.add_argument('--inference_ref_name', type=str, default='57541', help='which reference sample to inference')
        parser.add_argument('--inference_tag_name', type=str, default='56001', help='which target sample to inference')
        parser.add_argument('--inference_orient_name', type=str, default='56001', help='which orient sample to inference, if not specified, means use reference orient')
        parser.add_argument('--remove_background', action='store_true', help='if specified, remove background when output the fake image')
        parser.add_argument('--subset', type=str, default='val', help='which subset to test [val | train]')
        parser.add_argument('--expand_tag_mask', action='store_true', help='if specified, expand the tag hair mask before input..')
        parser.add_argument('--expand_th', type=int, default=11, help='expaned the tag hair mask for background encode.')
        parser.add_argument('--expand_mask_be', action='store_true', help='if sepcified, expaned the tag hair mask for background encode.')

        self.isTrain = False
        return parser
