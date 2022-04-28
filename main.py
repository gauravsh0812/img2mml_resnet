# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from train import train
from test import evaluate
from preprocessing.preprocess_img2mml import preprocess
from preprocessing.preprocess_images import preprocess_images
from model.model import Encoder, Decoder, Img2Seq
import time
import math
import argparse
import logging
import itertools

# import torcheck  # can be used to check if the model is working fine or not. Also,
# can be used to check if any of the parameter or weight is dying or exploding.

# argument
parser = argparse.ArgumentParser()
parser.add_argument( '--gpu_num', type=int, metavar='', required=True,
                            help='which gpu core want to use?')
args = parser.parse_args()

def define_model(SRC, TRG, DEVICE):#, TRG_PAD_IDX, OUTPUT_DIM):
    '''
    defining the model
    initializing encoder, decoder, and model
    '''

    print('defining model...')
    INPUT_CHANNEL = 3
    ENC_DIM = 2048
    OUTPUT_DIM = len(TRG.vocab)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    DEC_EMB_DIM = 256
    HID_DIM = 500
    N_LAYERS = 1
    DROPOUT = 0.5

    print('building model...')
    ENC = Encoder()
    DEC = Decoder(DEC_EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, ENC_DIM, DROPOUT)
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

batch_size = 128
best_valid_loss = float('inf')
device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

SRC, TRG, train_iter, test_iter, val_iter = preprocess(device, batch_size)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
model = define_model(SRC, TRG, device)#, TRG_PAD_IDX, output_dim)
model.to(device)

print('MODEL: ')
print(model.apply(init_weights))
print(f'The model has {count_parameters(model):,} trainable parameters')
# with open('logs/model.txt', 'w') as f:
#    f.write(model.apply(init_weights))

# optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

EPOCHS = 1
CLIP = 1

# to save trained model and logs
FOLDER = ['trained_models', 'logs']
for f in FOLDER:
    if not os.path.exists(f):
        os.mkdir(f)
# to log losses after every epoch
loss_file = open('logs/loss_file.txt', 'w')

for epoch in range(EPOCHS):

    start_time = time.time()
    train_loss = train(TRG, model, batch_size, train_iter, optimizer, criterion, device, CLIP, False) # No writing outputs
    val_loss = evaluate(TRG, model, batch_size, val_iter, criterion, device, True)
    end_time=time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), f'trained_models/opennmt-version1-model.pt')

    # logging
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

    loss_file.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
    loss_file.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
    loss_file.write(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n')



print('final model saved at:  ', f'trained_models/opennmt-version1-model.pt')

# testing
model.load_state_dict(torch.load(f'trained_models/opennmt-version1-model.pt'))
test_loss = evaluate(TRG, model, batch_size, test_iter, criterion, device, True)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
loss_file.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# stopping time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
