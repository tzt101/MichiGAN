"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import pdb


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            if not opt.unpairTrain:
                self.optimizer_G, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            else:
                self.optimizer_G, self.optimizer_D, self.optimizer_D2 = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        self.d_losses = {}
        self.nanCount = 0

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()

        # flag = False
        # for n, p in self.pix2pix_model.module.netB.named_parameters():
        #     g = p.grad
        #     if (g != g).sum() > 0:
        #         flag = True
        #         self.nanCount = self.nanCount + 1
        #         break
        # if self.nanCount > 100:
        #     pdb.set_trace()
        # if not flag:
        #     self.optimizer_G.step()
        # print('count:', self.nanCount)
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        if self.opt.curr_step == 1:
            # print('step1')
            self.optimizer_D.zero_grad()
            d_losses = self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D.step()
            self.d_losses = d_losses
        else:
            # print('step2')
            self.optimizer_D2.zero_grad()
            d_losses = self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D2.step()
            self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    def init_losses(self):
        self.g_losses = {}
        self.d_losses = {}

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
