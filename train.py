"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""
import torch

if not torch.set_flush_denormal(True):
    print("Unable to set flush denormal")
    print("Pytorch compiled without advanced CPU")
    print("at: https://github.com/pytorch/pytorch/blob/84b275b70f73d5fd311f62614bccc405f3d5bfa3/aten/src/ATen/cpu/FlushDenormal.cpp#L13")
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
if opt.unpairTrain:
    dataloader2 = data.create_dataloader(opt, 2)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))
data_size = len(dataloader)

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    # for unpair training
    if opt.unpairTrain:
        # dataloader2 = data.create_dataloader(opt, 2)
        iter_counter.record_epoch_start(epoch)
        opt.curr_step = 2
        trainer.init_losses()
        for i, data_i in enumerate(dataloader2, start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            if i % opt.G_steps_per_D == 0 and not opt.no_discriminator:
                trainer.run_discriminator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_ref', data_i['label_ref']),
                                       ('input_tag', data_i['label_tag']),
                                       ('synthesized_image', trainer.get_latest_generated()),
                                       ('image_ref', data_i['image_ref']),
                                       ('image_tag', data_i['image_tag'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()


        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        # if epoch % opt.save_epoch_freq == 0 or \
        #         epoch == iter_counter.total_epochs:
        #     print('saving the model at the end of epoch %d, iters %d' %
        #             (epoch, iter_counter.total_steps_so_far))
        #     trainer.save('latest')
        #     trainer.save(epoch)
    # for step 1 training
    # dataloader = data.create_dataloader(opt)
    iter_counter.record_epoch_start(epoch)
    opt.curr_step = 1
    trainer.init_losses()
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        if i % opt.G_steps_per_D == 0 and not opt.no_discriminator:
            trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_ref', data_i['label_ref']),
                                   ('input_tag', data_i['label_tag']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('image_ref', data_i['image_ref']),
                                   ('image_tag', data_i['image_tag'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

        # if (i + 1) * opt.batchSize >= 28000 and opt.unpairTrain:
        #     break

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)


print('Training was successfully finished.')
