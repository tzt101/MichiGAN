"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--G_steps_per_D', type=int, default=1, help='number of generator iterations per discriminator iterations.')

        # for progressive training
        parser.add_argument('--smooth', action='store_true', help='if specified, smooth the training between each resolution.')
        parser.add_argument('--epoch_each_step', type=int, default=10, help='number of epochs for each resolution.')

        # add unpair training
        parser.add_argument('--unpairTrain', action='store_true', help='if specified, use unpair training strategy.')
        parser.add_argument('--curr_step', type=int, default=1, help='point out the step [1|2], 1 means the pair training stage and 2 means the unpair training stage.')
        parser.add_argument('--same_netD_model', action='store_true', help='if specified, use the same model to init netD and netD2.')
        parser.add_argument('--lambda_hairavglab', type=float, default=1.0, help='weight for hair avg lab l1 loss')


        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=1.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for vgg loss')
        parser.add_argument('--lambda_orient', type=float, default=10.0, help='weight for orientation loss')
        parser.add_argument('--lambda_confidence', type=float, default=100.0, help='weight for confidence loss')
        parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
        parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
        parser.add_argument('--lambda_background', type=float, default=1.0, help='weight for background loss')
        parser.add_argument('--lambda_rgb', type=float, default=1.0, help='weight for rgb l1 loss')
        parser.add_argument('--lambda_lab', type=float, default=1.0, help='weight for lab l1 loss')
        parser.add_argument('--no_gan_loss', action='store_true', help='if specified, do *not* use GAN loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_background_loss', action='store_true', help='if specified, do *not* use background loss')
        parser.add_argument('--no_rgb_loss', action='store_true', help='if specified, do *not* use rgb l1 loss')
        parser.add_argument('--no_lab_loss', action='store_true', help='if specified, do *not* use lab l1 loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument('--no_orient_loss', action='store_true', help='if specified, do *not* use orient constraint loss')
        parser.add_argument('--no_confidence_loss', action='store_true',
                            help='if specified, do *not* use confidence constraint loss')
        parser.add_argument('--no_content_loss', action='store_true', help='if specified, do *not* use content loss')
        parser.add_argument('--no_style_loss', action='store_true', help='if specified, do *not* use style loss')
        parser.add_argument('--remove_background', action='store_true', help='if specified, remove background when calculate loss')
        parser.add_argument('--orient_filter', type=str, default='gabor', help='which filter is cal orient [gabor|dog]')
        parser.add_argument('--wide_edge', type=float, default=1.0, help='if value bigger than 1, highlight the wide edge weight when cal GAN loss')
        parser.add_argument('--no_discriminator', action='store_true', help='if specified, do *not* use discriminator')

        # Lab balance
        parser.add_argument('--balance_Lab', action='store_true', help='if specified, add weight when cal the Lab loss')
        parser.add_argument('--weight_dir', type=str, default='./data/ab_count.npy', help='weight file dir')
        parser.add_argument('--Lab_weight_th', type=float, default=10.0, help='The max weight value')

        self.isTrain = True
        # self.only_tag = True
        return parser
