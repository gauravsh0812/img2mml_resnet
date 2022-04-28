import numpy as np
import pandas as pd
import random
import torch
import torchvision
from torch.utils.data import Dataset, Dataloader
import os
from sklearn.model_selection import train_test_split
# from torchtext.legacy.data import Field, BucketIterator, TabularDataset
# import torchtext.vocab as vocab

# set up seed
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def split_save_dataset():
    # reading raw text files
    mml_txt = open('data/mml.txt').read().split('\n')[:-1]
    image_num = range(0,len(mml_txt))
    raw_data = {'ID': [f'{num}' for num in image_num],
                'MML': [mml for mml in mml_txt]}

    df = pd.DataFrame(raw_data, columns=['ID','MML'])
    train_val, test = train_test_split(df, test_size = 0.1)
    train, val = train_test_split(train_val, test_size=0.1)

    train.to_json('data/train_i2s.json', orient='records', lines=True)
    test.to_json('data/test_i2s.json', orient='records', lines=True)
    val.to_json('data/val_i2s.json', orient='records', lines=True)


class MMLDataset(Dataset):

    def __init__(self, FILE):
        data = np.loadtxt(FILE)

    def __getitem__(self, index, FILE):
        return FILE[index]

    def __len__(self):
        return



print('preprocessing data...')
split_save_dataset()
