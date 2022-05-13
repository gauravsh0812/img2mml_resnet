# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
import logging
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from train_ddp import train
from test_ddp import evaluate
from preprocessing.preprocess_dataloader import preprocess
from model.model import Encoder, Decoder, Img2Seq

# import torcheck  # can be used to check if the model is working fine or not. Also,
# can be used to check if any of the parameter or weight is dying or exploding.

# argument
parser = argparse.ArgumentParser()
''' FOR DDP
parser.add_argument( '--rank', type=int, metavar='', required=True,
                            help='which gpu core want to use?')
'''
parser.add_argument( '--gpu_num', type=int, metavar='', required=True,
                            help='which gpu core want to use?')
args = parser.parse_args()
''' FOR DDP
rank = args.rank
'''
rank = args.gpu_num

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ['CUDA_LAUNCH_BLOCKING']="1"

def define_model(vocab, DEVICE):#, TRG_PAD_IDX, OUTPUT_DIM):
    '''
    defining the model
    initializing encoder, decoder, and model
    '''

    print('defining model...')
    INPUT_CHANNEL = 3
    OUTPUT_DIM = len(vocab)
    ENC_DIM = 512
    ATTN_DIM = 512
    DEC_EMB_DIM = 256
    HID_DIM = 500
    N_LAYERS = 1
    DROPOUT = 0.5

    print('building model...')
    ENC = Encoder()
    DEC = Decoder(DEC_EMB_DIM, ENC_DIM,  HID_DIM, ATTN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    model = Img2Seq(ENC, DEC, DEVICE, ENC_DIM, HID_DIM)

    return model

def init_weights(m):
    '''
    initializing the model wghts with values
    drawn from normal distribution.
    else initialize them with 0.
    '''
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def setup(rank, world_size):
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # export GLOO_SOCKET_IFNAME=en0

    # initialize the process group
    dist.init_process_group("gloo", init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def count_parameters(model):
    '''
    counting total number of parameters
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    '''
    epoch timing
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_checkpoint(epoch, encoder, decoder):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param encoder: encoder model
    :param decoder: decoder model
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder}
    torch.save(state, 'trained_models/opennmt-version1-model.pt')

# parameters
EPOCHS = 1
CLIP = 1
batch_size = 128
best_valid_loss = float('inf')

'''  FOR DDP
# parameters needed for DDP:
world_size = torch.cuda.device_count()  # total number of GPUs
rank = rank                               # sequential id of GPU

print(f'DDP_Model running on rank: {rank}...')
setup(rank, world_size)
'''

# device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)
assert torch.cuda.current_device() == 1
# print(torch.cuda.is_available())


''' FOR DDP
train_dataloader, test_dataloader, val_dataloade, vocab = preprocess(device, batch_size, rank, world_size)
'''
train_dataloader, test_dataloader, val_dataloader, vocab = preprocess(device, batch_size)
TRG_PAD_IDX = 0     # can be obtained from vocab in preprocessing <pad>:0, <unk>:1, <sos>:2, <eos>:3
model = define_model(vocab, device)
model.to(device)

''' FOR DDP
# Wrap the model in DDP wrapper
ddp_model = DDP(model, device_ids=[rank], output_device=rank)

print('MODEL: ')
print(ddp_model.apply(init_weights))
print(f'The model has {count_parameters(ddp_model):,} trainable parameters')

'''

# print('MODEL: ')
# print(model.apply(init_weights))
# print(f'The model has {count_parameters(model):,} trainable parameters')
#
# print('output dim:  ', len(vocab))

# print('cuda available:  ', torch.cuda.is_available())
# print('current device:  ', torch.cuda.current_device())

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# to save trained model and logs
FOLDER = ['trained_models', 'logs']
for f in FOLDER:
    if not os.path.exists(f):
        os.mkdir(f)
# to log losses after every epoch
loss_file = open('logs/loss_file.txt', 'w')

for epoch in range(EPOCHS):

    start_time = time.time()

    # train_dataloader.sampler.set_epoch(epoch)
    ''' FOR DDP
    train_loss = mp.spawn(train, args= (ddp_model, batch_size, train_dataloader, optimizer, criterion, device, CLIP, False), nprocs=world_size, join=True)
    train_loss = train(ddp_model, batch_size, train_dataloader, optimizer, criterion, device, CLIP, False) # No writing outputs
    val_loss = evaluate(ddp_model, batch_size, val_dataloader, criterion, device, True)
    '''
    train_loss, encoder, decoder = train(model, vocab, batch_size, train_dataloader, optimizer, criterion, device, CLIP, False) # No writing outputs
    val_loss, encoder, decoder = evaluate(model, vocab, batch_size, val_dataloader, criterion, device, True)
    end_time=time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        #if rank == 0:
        torch.save(model.state_dict(), f'trained_models/opennmt-version1-model.pt')
        # save_checkpoint(epoch, encoder, decoder)

    # logging
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

    loss_file.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
    loss_file.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
    loss_file.write(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n')

    ''' FOR DDP
    cleanup()
    '''

print('final model saved at:  ', f'trained_models/opennmt-version1-model.pt')

# testing
''' FOR DDP
ddp_model.load_state_dict(torch.load(f'trained_models/opennmt-version1-model.pt'))
test_loss = evaluate(ddp_model, batch_size, test_dataloader, criterion, device, True)
'''
# model = torchvision.models.resnet18(pretrained=True)

# model.load_state_dict(checkpoint['state_dict'], strict=False)
model.load_state_dict(torch.load(f'trained_models/opennmt-version1-model.pt'))#, strict=False)
test_loss, encoder, decoder = evaluate(model, vocab, batch_size, test_dataloader, criterion, device, True)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
loss_file.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# stopping time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
