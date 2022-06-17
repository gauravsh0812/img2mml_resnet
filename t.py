#import torch
#torch.set_printoptions(profile="full")
#print(torch.load('data/image_tensors/1.txt'))

import os, shutil
mini_mml = open('data/mml.txt', 'w')
mml = open('data_full/mml.txt').readlines()[:1000]
for i, m in enumerate(mml):
    mini_mml.write(m)
    shutil.copyfile(f'data_full/image_tensors/{i}.txt', f'data/image_tensors/{i}.txt')
    
