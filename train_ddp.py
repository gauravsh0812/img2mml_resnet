# -*- coding: utf-8 -*-

import time
import torch
from preprocessing.preprocess_images import preprocess_images

def train(model, batch_size, train_dataloader, optimizer, criterion,device, clip, write_file):

    model.train()  # train mode is ON i.e. dropout and normalization tech. will be used

    epoch_loss = 0

    trg_seqs = open('logs/train_targets.txt', 'w')
    pred_seqs = open('logs/train_predicted.txt', 'w')

    for img, mml in enumerate(iter(train_dataloader)):

        # if i%10==0: print(i)
        # initailize the hidden state
        trg = mml.to(device)
        print('@train trg shape:  ', trg.shape)
        batch_size = trg.shape[1]

        # grab the image and preprocess it
        print('@train img shape:  ', img.shape)
        src = preprocess_images(img, 'data/images/')
        # src will be list of image tensors
        # need to pack them to create a single batch tensor
        src = torch.stack(src).to(device)

        # setting gradients to zero
        optimizer.zero_grad()

        output, pred = model(trg_field, src, trg, True, True, 0.5)

        # translating and storing trg and pred sequences in batches
        if write_file:
            batch_size = trg.shape[1]
            for idx in range(batch_size):
                trg_arr = [trg_field.vocab.itos[itrg] for itrg in trg[:,idx]] #[trg_vocab_itos[itrg] for itrg in trg[:,idx]] #
                trg_seq = " ".join(trg_arr)
                trg_seqs.write(trg_seq + '\n')

                pred_arr = [trg_field.vocab.itos[ipred] for ipred in pred[:,idx].int()] #[trg_vocab_itos[ipred] for ipred in pred[:,idx].int()]#
                pred_seq = " ".join(pred_arr)
                pred_seqs.write(pred_seq+'\n')

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        #print('trg2: ', trg.requires_grad)
        #print(output.requires_grad)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(iterator)
