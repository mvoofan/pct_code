import os
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from functools import cmp_to_key
import re


def my_compare(x,  y):
    return int( re.findall(r'\d+',x.split('/')[-1] ) )-int( re.findall(r'\d+',y.split('/')[-1] ) )



class MyAlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths=os.listdir(self.dir_AB)
        if opt.phase is not 'train':
            self.AB_paths=sorted(self.AB_paths, key=cmp_to_key(lambda x,  y:int(  re.findall(r'\d+',x.split('/')[-1] )[0]  )-int(  re.findall(r'\d+',y.split('/')[-1] )[0]  ) ) )


        assert(opt.resize_or_crop == 'resize_and_crop')

        '''
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        '''
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = os.path.join( self.dir_AB, self.AB_paths[index]  )
        ttt=np.fromfile(AB_path, dtype=np.float32)
        A=ttt[0:8*264*264*2].reshape((8,264,264,2))
        B=ttt[8*264*264*2:8*264*264*4].reshape((8,264,264,2))

        A=A.transpose((3,0,1,2))
        B=B.transpose((3,0,1,2))



        '''remove to add the z axis layer channel
        A=A[0,:,:,:]
        A=A[np.newaxis,...]
        '''
        
        B=B[0,:,:,:]
        B=B[np.newaxis,...]
        B=B*0.1

        '''
        A=np.tile(A, (1,16,1,1))    #for debug
        B=np.tile(B, (1,16,1,1))    #for debug
        '''
        A=A[:,:,4:260,4:260]
        B=B[:,:,4:260,4:260]



        '''
        A=self.transform(A)
        B=self.transform(B)
        '''
        A=torch.from_numpy(A)
        B=torch.from_numpy(B)





        '''
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
        '''

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'MyAlignedDataset'
