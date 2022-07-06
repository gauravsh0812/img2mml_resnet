# creating tab dataset having image_num and mml
# split train, test, val
# dataloader to load data
import numpy as np
import pandas as pd
import random
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
#from torchtext.legacy.vocab import Vocab
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from functools import partial


class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        eqn = self.dataframe.iloc[index, 1]
        indexed_eqn = []
        for token in eqn.split():
            if self.vocab[token] != None:
                indexed_eqn.append(self.vocab[token])
            else:
                indexed_eqn.append(self.vocab['<unk>'])

        return self.dataframe.iloc[index, 0],torch.Tensor(indexed_eqn)

# class Img2MML_dataset(Dataset):
#     def __init__(self, dataframe, vocab, tokenizer):
#         self.dataframe = dataframe
#
#         for l in range(len(self.dataframe)):
#             eqn = self.dataframe.iloc[l, 1]
#             indexed_eqn = []
#             for token in tokenizer(eqn):
#                 if vocab[token] != None:
#                     indexed_eqn.append(vocab[token])
#                 else:
#                     indexed_eqn.append(vocab['<unk>'])
#
#             self.dataframe.iloc[l, 1] = torch.Tensor(indexed_eqn)
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, index):
#         return self.dataframe.iloc[index, 0], self.dataframe.iloc[index, 1]

class My_pad_collate(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        _img, _mml = zip(*batch)

        # padding
        padded_mml_tensor = pad_sequence(_mml, padding_value=0)
        _img = [i for i in _img]


        #return torch.Tensor(_img).to(self.device), padded_mml_tensor.to(self.device)
        return torch.stack(_img).to(self.device), padded_mml_tensor.to(self.device)


def preprocess(device, batch_size, args_arr):

    print('preprocessing data...')

    if len(args_arr) == 2:
        args_arr = rank, world_size
        ddp = True
    else:
        ddp = False

    print('ddp: ', ddp)
    # reading raw text files
    mml_txt = open('data/mml.txt').read().split('\n')[:-1]
    image_num = range(0,len(mml_txt))

    # adding <sos> and <eos> tokens then creating a dataframe
    # raw_mml_data = {'ID': [f'{num}' for num in image_num],
    #                 'MML': [('<sos> '+ mml + ' <eos>') for mml in mml_txt]}
    raw_mml_data = {'ID': [torch.load(f'data/image_tensors/{num}.txt') for num in image_num],
                    'MML': [('<sos> '+ mml + ' <eos>') for mml in mml_txt]}

    df = pd.DataFrame(raw_mml_data, columns=['ID','MML'])

    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)

    # sort train dataset
    train = train.sort_values(by='MML', key=lambda x: x.str.len())

    # build vocab
    counter = Counter()
    for line in train['MML']:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    vocab = Vocab(counter, min_freq=10, specials=['<pad>', '<unk>', '<sos>', '<eos>'])

    # writing vocab file...
    vfile = open('logs/vocab.txt', 'w')
    for vidx, vstr in vocab.stoi.items():
        vfile.write(f'{vidx} \t {vstr} \n')

    # define tokenizer function
    tokenizer = lambda x: x.split()

    train.to_csv('data/train_i2s.csv', index=False)
    test.to_csv('data/test_i2s.csv', index=False)
    val.to_csv('data/val_i2s.csv', index=False)

    # Creating copy of datasets to avoid "SettingWithCopyWarning"
    train_copy = train.copy()
    test_copy = test.copy()
    val_copy = val.copy()

    # initializing pad collate class
    mypadcollate = My_pad_collate(device)

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train_copy,
                                 vocab,
                                 tokenizer)
    '''    FOR DDP '''
    if ddp:
        # Create distributed sampler pinned to rank
        train_sampler = DistributedSampler(imml_train,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=True,  # May be True
                                     seed=42)
        train_dataset = train_sampler
    else:
        train_dataset = imml_train

    # creating dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=0,
                                  shuffle=True,
                                  collate_fn=mypadcollate,
                                  pin_memory=False)

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test_copy,
                                vocab,
                                tokenizer)

    test_dataloader = DataLoader(imml_test,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=True,
                                 collate_fn=mypadcollate,
                                 pin_memory=False)

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val_copy,
                               vocab,
                               tokenizer)

    val_dataloader = DataLoader(imml_val,
                                batch_size=batch_size,
                                num_workers=0,
                                shuffle=True,
                                collate_fn=mypadcollate,
                                pin_memory=False)

    return train_dataloader, test_dataloader, val_dataloader, vocab
