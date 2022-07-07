# -*- coding: utf-8 -*-

import os, random
import numpy as np
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
# from model.model import Encoder, Decoder, Img2Seq
from model.cnn_encoder import OpenNMTEncoder, OpenNMTDecoder, OpenNMTImg2Seq

# import torcheck  # can be used to check if the model is working fine or not. Also,
# can be used to check if any of the parameter or weight is dying or exploding.

# argument
parser = argparse.ArgumentParser()
''' FOR DDP '''
parser.add_argument( '--local_rank', type=int, metavar='', required=False, default=0,
                            help='which gpu core want to use?')
# parser.add_argument("--ddp", default=True, action="store_true",
#                     help="should run in DDP mode or single GPU")
parser.add_argument( '--batch_size', type=int, metavar='', required=True,
                            help='Batch size')
parser.add_argument( '--epochs', type=int, metavar='', required=True,
                            help='number of epochs')
args = parser.parse_args()
ddp = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# export CUDA_VISIBLE_DEVICES=0,1
# os.environ['CUDA_LAUNCH_BLOCKING']="1"

def set_random_seed(seed):
    # set up seed
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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
    HID_DIM = 512
    N_LAYERS = 1
    DROPOUT = 0.3

    print('building model...')
    # ENC = Encoder()
    # DEC = Decoder(DEC_EMB_DIM, ENC_DIM,  HID_DIM, ATTN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    # model = Img2Seq(ENC, DEC, DEVICE, ENC_DIM, HID_DIM)

    ENC = OpenNMTEncoder(INPUT_CHANNEL, HID_DIM, N_LAYERS, DROPOUT, DEVICE)
    DEC = OpenNMTDecoder(DEC_EMB_DIM, ENC_DIM,  HID_DIM, ATTN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    model = OpenNMTImg2Seq(ENC, DEC, DEVICE, ENC_DIM, HID_DIM)


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
    os.environ['MASTER_PORT'] = '1234'

    # export GLOO_SOCKET_IFNAME=en0
    print('setting up environ...')

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
EPOCHS = args.epochs
CLIP = 1
batch_size = args.batch_size
best_valid_loss = float('inf')
rank = args.local_rank           # sequential id of GPU

device = torch.device(f'cuda'if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(rank)
# assert torch.cuda.current_device() == rank

# set_random_seed
set_random_seed(seed=42)

'''  FOR DDP '''
# if args.ddp:
#     # parameters needed for DDP:
#     world_size = torch.cuda.device_count()  # total number of GPUs
#
#     print(f'DDP_Model running on rank: {rank}...')
#     setup(rank, world_size)

# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
if ddp: torch.distributed.init_process_group(backend="nccl")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

''' FOR DDP '''
if ddp:#args.ddp:
    train_dataloader, test_dataloader, val_dataloade, vocab = preprocess(device, batch_size, [rank, world_size])
else:
    train_dataloader, test_dataloader, val_dataloader, vocab = preprocess(device, batch_size, [])

TRG_PAD_IDX = 0     # can be obtained from vocab in preprocessing <pad>:0, <unk>:1, <sos>:2, <eos>:3
model = define_model(vocab, device)
model = nn.DataParallel(model.cuda(), device_ids=[0, 1])
# model.to(device)


''' FOR DDP '''
if ddp:#args.ddp:
    # Wrap the model in DDP wrapper
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    model = ddp_model


print('MODEL: ')
print(model.apply(init_weights))
print(f'The model has {count_parameters(model):,} trainable parameters')

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# to save trained model and logs
# it is not a good practice to to create directories while using DDP
if not ddp:#args.DDP:
    FOLDER = ['trained_models', 'logs']
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

# to log losses after every epoch
loss_file = open('logs/loss_file.txt', 'w')

for epoch in range(EPOCHS):

    start_time = time.time()
    write_train_file_flag = False
    write_val_file_flag = False
    if epoch%50==0:
        write_train_file_flag = True
        write_val_file_flag = True
    # train_dataloader.sampler.set_epoch(epoch)
    ''' FOR DDP '''
    if ddp:#args.ddp:
        train_loss = mp.spawn(train, args= (ddp_model, batch_size, train_dataloader, optimizer, criterion, device, CLIP, False), nprocs=world_size, join=True)
        val_loss = mp.spawn(evaluate, args=(ddp_model, batch_size, val_dataloader, criterion, device, True), nprocs=world_size, join=True)
    else:
        #train_loss, encoder, decoder = train(model, vocab, batch_size, train_dataloader, optimizer, criterion, device, CLIP, False) # No writing outputs
        train_loss = train(model, epoch, vocab, batch_size, train_dataloader, optimizer, criterion, device, CLIP, write_train_file_flag) # No writing outputs
        #val_loss, encoder, decoder = evaluate(model, vocab, batch_size, val_dataloader, criterion, device, True)
        val_loss = evaluate(model, epoch, vocab, batch_size, val_dataloader, criterion, device, write_val_file_flag)

    end_time=time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        if rank == 0:
            torch.save(model.state_dict(), f'trained_models/opennmt-version1-model.pt')
        # save_checkpoint(epoch, encoder, decoder)

    # logging
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

    loss_file.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
    loss_file.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
    loss_file.write(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n')

    ''' FOR DDP     '''
    if ddp:#args.ddp:
        cleanup()


print('final model saved at:  ', f'trained_models/opennmt-version1-model.pt')

# testing
model.load_state_dict(torch.load(f'trained_models/opennmt-version1-model.pt'))

''' FOR DDP '''
if ddp:#args.ddp:
    test_loss = mp.spawn(evaluate, args=(ddp_model, batch_size, test_dataloader, criterion, device, True), nprocs=world_size, join=True)
else:
    #test_loss, encoder, decoder = evaluate(model, vocab, batch_size, test_dataloader, criterion, device, True)
    epoch = 'test_0'
    test_loss = evaluate(model, epoch, vocab, batch_size, test_dataloader, criterion, device, True)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
loss_file.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# stopping time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
