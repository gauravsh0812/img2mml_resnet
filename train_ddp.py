# -*- coding: utf-8 -*-

import time, pandas
import torch
import json

def train(model, epoch, vocab, batch_size, train_dataloader, optimizer, criterion, device, clip, write_file):

    model.train()  # train mode is ON i.e. dropout and normalization tech. will be used

    # train_dataloader.sampler.set_epoch(i)

    epoch_loss = 0

    # write the target file once as it will be same for every epoch
    if epoch==0: trg_seqs = open(f'logs/train_targets_100K.txt', 'w')

    # opening predicted file
    if write_file:
        all_preds = []
        pred_seqs = open(f'logs/train_predicted_100K_epoch_{epoch}.txt', 'w')

    for i, (img, mml) in enumerate(train_dataloader):
    # for i, tdi in enumerate(train_dataloader):
        #print(f'============== {i} ================')
        # trg = trg.permute(1,0)    # trg: [len, B] --> [B, len]
        trg = mml
        trg = trg.permute(1,0)    # trg: [len, B] --> [B, len]
        # trg.to(device, dtype=torch.int64)
        trg.to(device, dtype=torch.int64)
        # trg = mml.to(device, dtype=torch.int64)
        batch_size = trg.shape[0]
        # print('train batch: ', batch_size.shape)

        # loading image Tensors
        # srcTensor = []
        # for _i in img:
        #    srcTensor.append(torch.load(f'data/image_tensors/{int(_i)}.txt'))
        # src = (torch.stack(srcTensor)).to(device)

        # src = img.to(device)
        src = img
        src.to(device)

        # print('trg_shape: ', trg.shape)
        # print('src shape:  ', src.shape)

        # setting gradients to zero
        optimizer.zero_grad()

        #output, pred, encoder, decoder = model(src, trg, vocab, True, True, 0.5)
        # pred --> [B, len]
        output, pred = model(src, trg, vocab, True, True, 0.5)
        # output, pred, encoder, decoder = model( tdi, vocab, True, True, 0.5 )
        output = output.permute(1,0,2)

        # translating and storing trg and pred sequences in batches
        # writing target eqns
        if epoch==0:
            for idx in range(batch_size):
                trg_arr = [vocab.itos[itrg] for itrg in trg.int()[idx,:]]
                trg_seq = " ".join(trg_arr)
                trg_seqs.write(trg_seq + '\n')

        if write_file:
            all_preds.append(pred)
            for idx in range(batch_size):
                pred_arr = [vocab.itos[ipred] for ipred in pred.int()[idx,:]]
                pred_seq = " ".join(pred_arr)
                pred_seqs.write(pred_seq+'\n')

        #trg = [trg len, batch size]
        trg = trg.permute(1,0)
        #output = [trg len, batch size, output dim]
        #print('output permute size: ', output.shape)
        #print('trg:  ', trg.shape)
        #print('output view size: ', output[1:].contiguous().view(-1, output.shape[-1]).shape)
        #print('trg view shape: ', trg[1:].view(-1).shape)

        #output = [B, trg len, output dim] --> [len, B, out]
        # print(trg.shape,  output.shape)
        output_dim = output.shape[-1]
        output = output[1:].contiguous().view(-1, output_dim)
        trg = trg[1:].view(-1)
        #print('trg dtype: ', trg.dtype)
        #print('output dtype: ', output.dtype)
        #print('trg:  ', trg.to(torch.int64))
        #print('output: ', output.to(torch.int64))
        # print('training scripts:  ', output.shape,  trg.shape)
        loss = criterion(output, trg.to(torch.int64))
        # print('output size: ', output.shape)
        # print('trg:  ', trg.shape)
        # print('loss:  ', loss.shape)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        net_loss = epoch_loss/len(train_dataloader)

    #return net_loss, encoder, decoder
    if write_file:
        json.dump(all_preds, open(f'logs/preds_epoch_{epoch}.txt', 'w'), indent=4)

    return net_loss
