import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as  np
import torch
from put_data_in_test import put_data_in_test
from combine_image import combine_image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

'''
python test.py --dataroot /yaop/raofan/3d-pix2pix-CycleGAN-master/dataset/ 
'''
opt = TestOptions().parse()

opt.dataroot=r'/yaop/raofan/3d-pix2pix-CycleGAN-master/dataset/'
opt.phase='test_data_expandded'
opt.loadSize=256
opt.fineSize=256
opt.depthSize=8
opt.input_nc=2
opt.output_nc=1
opt.which_model_netD='basic'
opt.which_model_netG='unet_copy'
opt.dataset_mode='myaligned'
opt.model='pix2pix3d'
#opt.batchSize=1
opt.gpu_ids = '0' 
opt.load_G_net = '/yaop/raofan/3d-pix2pix-CycleGAN-master_v2/checkpoints_6/experiment_name/latest_net_G_Jul25-1544.pth'   # to be added
opt.load_D_net =  '/yaop/raofan/3d-pix2pix-CycleGAN-master_v2/checkpoints_6/experiment_name/latest_net_D_Jul25-1544.pth'  # to be added
opt.results_dir ='./results/'
opt.nThreads=1


'''
self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
'''
'''
            "--load_G_net","/yaop/raofan/3d-pix2pix-CycleGAN-master_v2/checkpoints_5/experiment_name/latest_net_G_Jul19-1046.pth",
            "--load_D_net","/yaop/raofan/3d-pix2pix-CycleGAN-master_v2/checkpoints_5/experiment_name/latest_net_D_Jul19-1046.pth"
'''
opt.how_many=23
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
print('opt:')
#print(opt)
abc=vars(opt)
for k, v in sorted(abc.items()):
    print('%s: %s' % (str(k), str(v)))


################################

'''
prompt_file=['No.9LFY_Bed3_prompt_bp.bin']
prompt_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct_data/pp_output_testing/',file) for file in prompt_file]

random_file=['No.9LFY_Bed3_random_bp.bin']
random_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct_data/rd_output_testing/',file) for file in random_file]
'''

'''
prompt_file=['10_04_2018_delayed_scan_sino_pp_bed1_prompt_bp.bin', '10_04_2018_delayed_scan_sino_pp_bed2_prompt_bp.bin', '10_04_2018_normal_scan_sino_pp_bed1_prompt_bp.bin', '10_04_2018_normal_scan_sino_pp_bed2_prompt_bp.bin', '10_04_2018_normal_scan_sino_pp_bed3_prompt_bp.bin', '10_04_2018_normal_scan_sino_pp_bed4_prompt_bp.bin', '10_04_2018_normal_scan_sino_pp_bed5_prompt_bp.bin', '10_18_2018_delayed_scan_sino_pp_bed1_prompt_bp.bin', '10_18_2018_delayed_scan_sino_pp_bed2_prompt_bp.bin', '10_18_2018_normal_scan_sino_pp_bed1_prompt_bp.bin', '10_18_2018_normal_scan_sino_pp_bed2_prompt_bp.bin', '10_18_2018_normal_scan_sino_pp_bed3_prompt_bp.bin', '10_18_2018_normal_scan_sino_pp_bed4_prompt_bp.bin', '10_18_2018_normal_scan_sino_pp_bed5_prompt_bp.bin']
prompt_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct/mingfeng_bp_data/pp_output/',file) for file in prompt_file]


random_file=['10_04_2018_delayed_scan_Bed1_sino_de_random_bp.bin', '10_04_2018_delayed_scan_Bed2_sino_de_random_bp.bin', '10_04_2018_normal_scan_Bed1_sino_de_random_bp.bin', '10_04_2018_normal_scan_Bed2_sino_de_random_bp.bin', '10_04_2018_normal_scan_Bed3_sino_de_random_bp.bin', '10_04_2018_normal_scan_Bed4_sino_de_random_bp.bin', '10_04_2018_normal_scan_Bed5_sino_de_random_bp.bin', '10_18_2018_delayed_scan_Bed1_sino_de_random_bp.bin', '10_18_2018_delayed_scan_Bed2_sino_de_random_bp.bin', '10_18_2018_normal_scan_Bed1_sino_de_random_bp.bin', '10_18_2018_normal_scan_Bed2_sino_de_random_bp.bin', '10_18_2018_normal_scan_Bed3_sino_de_random_bp.bin', '10_18_2018_normal_scan_Bed4_sino_de_random_bp.bin', '10_18_2018_normal_scan_Bed5_sino_de_random_bp.bin']
random_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct/mingfeng_bp_data/rd_output/',file) for file in random_file]
'''

prompt_file=['10_18_2018_delayed_scan_sino_pp_bed2_prompt_bp.bin']
prompt_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct/mingfeng_bp_data/pp_output/',file) for file in prompt_file]


random_file=['10_18_2018_delayed_scan_Bed2_sino_de_random_bp.bin']
random_bp_list=[os.path.join(r'/yaop/raofan/bp_to_ct/mingfeng_bp_data/rd_output/',file) for file in random_file]


data_dir=os.path.join(opt.dataroot)

put_data_in_test(data_dir, prompt_bp_list, random_bp_list)    # to be added

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


model = create_model(opt)
#visualizer = Visualizer(opt)
# create website
#web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

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

'''
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
'''


if not os.path.exists(os.path.join(opt.results_dir,'evaled')):
    os.makedirs(os.path.join(opt.results_dir,'evaled'))

#torch.cuda.set_device(  int(opt.gpu_ids[0]) )    #added



for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    model.set_input(data)
    model.test()
    #visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    #print('process image... %s' % img_path)
    print('process image... %s' % model.image_paths[0])

    fake_B_np=model.fake_B.data.cpu().numpy()

    #out=np.pad(fake_B_np[0,0,:,:,:],((0,0),(4,4),(4,4) ),'constant', constant_values=(0,0,0) )
    out=np.zeros( (8,264,264) )
    out[:,4:260,4:260]=fake_B_np[0,0,:,:,:]
    out=out.astype(np.float32)

    bi_file = open(os.path.join(opt.results_dir,'evaled',('result%d.bin'    %    i) ), 'wb')
    bi_file.write(out.copy(order='C'))
    bi_file.close()
