from .base_options import BaseOptions


class DemoOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.add_argument('--which_epoch', type=str, default='50', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--expand_th', type=int, default=5, help='threshold of expaned the tag hair mask for background encode.')
        parser.add_argument('--expand_mask_be', action='store_true', help='if sepcified, expaned the tag hair mask for background encode.')

        parser.set_defaults(preprocess_mode='scale_width_and_crop')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(gpu_ids='0')
        parser.set_defaults(netG='spadeb')
        parser.set_defaults(use_encoder=True)
        parser.set_defaults(use_ig=True)
        parser.set_defaults(noise_background=True)
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(use_stroke=True)

        parser.set_defaults(name='MichiGAN')
        parser.set_defaults(expand_mask_be=True)
        parser.set_defaults(which_epoch='50')
        parser.set_defaults(add_feat_zeros=True)

        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1, gpu_ids='1')
        parser.add_argument('--demo_data_dir', type=str, default='./datasets/FFHQ_demo/')
        parser.add_argument('--results_dir', type=str,
                            default='/mnt/lvdisk1/tzt/HairSynthesis/SPADE-master/results/SPADEBEncodeInpaint5B/interactive_results/',
                            help='saves results here.')
        # parser.add_argument('--inference_ref_name', type=str, default='56001', help='which reference sample to inference')
        # parser.add_argument('--inference_tag_name', type=str, default='56001', help='which target sample to inference')
        # parser.add_argument('--inference_orient_name', type=str, default='56001', help='which orient sample to inference, if not specified, means use reference orient')
        # parser.add_argument('--remove_background', action='store_true', help='if specified, remove background when output the fake image')
        # parser.add_argument('--subset', type=str, default='val', help='which subset to test [val | train]')
        parser.add_argument('--expand_tag_mask', action='store_true', help='if specified, expand the tag hair mask before input.')


        self.isTrain = False
        return parser