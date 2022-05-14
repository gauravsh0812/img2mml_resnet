# -*- coding: utf-8 -*-

import time, pandas
import torch
# from preprocessing.preprocess_images import preprocess_images

def train(model, vocab, batch_size, train_dataloader, optimizer, criterion,device, clip, write_file):

    model.train()  # train mode is ON i.e. dropout and normalization tech. will be used

    epoch_loss = 0

    trg_seqs = open('logs/train_targets.txt', 'w')
    pred_seqs = open('logs/train_predicted.txt', 'w')
    
    print('len train dataloader: ', len(train_dataloader))
    flag600=False
    for i, (img, mml) in enumerate(train_dataloader):

        if i>600: 
            print(f'train_{i}')
            flag600 = True

        trg = mml.to(device, dtype=torch.int64)
        batch_size = trg.shape[1]

        # loading image Tensors
        srcTensor = []
        for _i in img:
            srcTensor.append(torch.load(f'data/image_tensors/{int(_i)}.txt'))
        src = torch.stack(srcTensor).to(device)

        #print('trg_shape: ', trg.shape)
        #print('src shape:  ', src.shape)

        # setting gradients to zero
        optimizer.zero_grad()

        output, pred, encoder, decoder = model(src, trg, vocab, True, True, flag600, 0.5)

        # translating and storing trg and pred sequences in batches
        if write_file:
            batch_size = trg.shape[1]
            for idx in range(batch_size):
                trg_arr = [vocab.itos[itrg] for itrg in trg[:,idx]]
                trg_seq = " ".join(trg_arr)
                trg_seqs.write(trg_seq + '\n')

                pred_arr = [vocab.itos[ipred] for ipred in pred.int()[:,idx]]
                pred_seq = " ".join(pred_arr)
                pred_seqs.write(pred_seq+'\n')

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        net_loss = epoch_loss/len(train_dataloader)

    return net_loss, encoder, decoder
