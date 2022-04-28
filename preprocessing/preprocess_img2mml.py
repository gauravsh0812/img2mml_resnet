# creating tab dataset having image_num and mml
# split train, test, val
# dataloader to load data
import numpy as np
import pandas as pd
import random
import torch
import os
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import torchtext.vocab as vocab

# set up seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def preprocess(device, batch_size):

    print('preprocessing data...')
    
    # reading raw text files
    mml_txt = open('data/mml.txt').read().split('\n')[:-1]
    image_num = range(0,len(mml_txt))
    raw_data = {'ID': [f'{num}' for num in image_num],
                'MML': [mml for mml in mml_txt]}

    df = pd.DataFrame(raw_data, columns=['ID','MML'])

    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)

    # train.to_csv('data/train_i2s.csv', index=False)
    # test.to_csv('data/test_i2s.csv', index=False)
    # val.to_csv('data/val_i2s.csv', index=False)

    train.to_json('data/train_i2s.json', orient='records', lines=True)
    test.to_json('data/test_i2s.json', orient='records', lines=True)
    val.to_json('data/val_i2s.json', orient='records', lines=True)
    '''
    # building json file for IM2LATEX-100K
    count = 0
    import pandas as pd
    for t in ['test', 'train','val']:
        src_t = open(f'data/src-{t}.txt').read().split('\n')[:-1]
        tgt_t = open(f'data/tgt-{t}.txt').read().split('\n')[:-1]
        # raw_data = {}
        # for num in range(len(src_t)):
        #     raw_data[str(count)] = tgt_t[num]
        #     os.rename(f'data/images/{src_t[num]}', f'data/images/{count}.png')
        #     count+=1
        raw_data = {'ID': [f'{num}' for num in src_t],
                    'LATEX': [mml for mml in tgt_t]}
        df = pd.DataFrame(raw_data, columns=['ID','LATEX'])

        df.to_json(f'data/{t}_i2s.json', orient='records', lines=True)
    '''
    # setting Fields
    # tokenizer will going be default tokenizer i.e. split by spaces
    SRC = Field(sequential=False,
                use_vocab=False
                )
    TRG = Field(
                init_token = '<sos>',
                eos_token = '<eos>',
                fix_length = 100
                )  # removing fixed length as Bucket iterator will take care of it.

    fields = {'ID':('id', SRC),'MML': ('mml', TRG)}
    #fields = {'ID':('id', SRC),'LATEX': ('mml', TRG)}
    train_data, test_data, val_data = TabularDataset.splits(
          path = 'data/',
          train = 'train_i2s.json',
          validation = 'val_i2s.json',
          test = 'test_i2s.json',
          format     = 'json',
          fields     = fields)

    #print(val_data[0].__dict__)
    # building vocab
    TRG.build_vocab(train_data, min_freq = 10)

    # writing vocab file
    with open('data/vocab.txt', 'w+') as f:
        for token, index in TRG.vocab.stoi.items():
            f.write(f'{index}\t{token}\n')


    # Iterator
    train_iter, test_iter, val_iter = BucketIterator.splits(
            (train_data, test_data, val_data),
            device = device,
            batch_size = batch_size,
            sort_within_batch = False,  # true if using pack_padded_sequences
            sort_key = lambda x: len(x.mml))

    #print(val_iter.dataset[0].__dict__)
    return SRC, TRG, train_iter, test_iter, val_iter
