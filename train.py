import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import os
import torch

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
#visualizer = Visualizer(opt)
total_steps = 0

data_num=0

try:
    opt.load_G_net
except NameError:
    print('Error in load_G_net')
else:
    if 'empty'==opt.load_G_net:
        print('emtpy in load_G_net')
    else:
        model.netG.load_state_dict(torch.load(opt.load_G_net))
        model.netG.eval()


try:
    opt.load_D_net
except NameError:
    print('Error in load_D_net')
else:
    if 'empty'==opt.load_D_net:
        print('emtpy in load_D_net')
    else:
        model.netD.load_state_dict(torch.load(opt.load_D_net))
        model.netD.eval()




for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    if 'pbar' in vars():
        try:
            pbar.close()
        except:
            print('error!')
    
    pbar = tqdm(total=dataset_size)

    for i, data in enumerate(dataset):
        pbar.update(1)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)

        if total_steps % 100 ==0:
            #show loss value
            print('loss_D:    %f    loss_G:    %f    D_learning_rate:    %f    G_learning_rate    %f  '  %  (model.loss_D.data.cpu().numpy(), model.loss_G.data.cpu().numpy(), model.optimizer_D.param_groups[0]['lr'],model.optimizer_G.param_groups[0]['lr']) )
            #print('loss_G   ', model.loss_G.data.cpu().numpy())



        model.optimize_parameters()

        '''
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
        '''

        '''
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        '''


            

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        if total_steps % (500* opt.batchSize) ==0:
            input_A_np=model.input_A.data.cpu().numpy()
            fake_B_np=model.fake_B.data.cpu().numpy()
            input_B_np=model.input_B.data.cpu().numpy()
            np.save(os.path.join(opt.checkpoints_dir, 'bp_step%d' % (data_num*100) ), input_A_np[:,0,:,:,:])
            np.save(os.path.join(opt.checkpoints_dir, 'output%d' % (data_num*100) ), fake_B_np[:,0,:,:,:])
            np.save(os.path.join(opt.checkpoints_dir, 'gt_ct%d' % (data_num*100)  ), input_B_np[:,0,:,:,:])
            data_num+=1

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
