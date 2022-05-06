# -*- coding: utf-8 -*-

import torch
from preprocessing.preprocess_images import preprocess_images

def evaluate(model, vocab, batch_size, iterator, criterion, device, write_file):

    model.eval()

    epoch_loss = 0

    trg_seqs = open('logs/test_targets.txt', 'w')
    pred_seqs = open('logs/test_predicted.txt', 'w')

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            # if i%10==0: print(i)
            # initailize the hidden state
            trg = mml.to(device)
            print('@test trg shape:  ', trg.shape)
            batch_size = trg.shape[1]

            # grab the image and preprocess it
            print('@test img shape:  ', img.shape)
            src = preprocess_images(img, 'data/images/')
            # src will be list of image tensors
            # need to pack them to create a single batch tensor
            src = torch.stack(src).to(device)


            # trg = batch.mml.to(device)

            output, pred = model(trg_field, src, trg, True)   # turn off teacher_forcing

            # translating and storing trg and pred sequences in batches
            if write_file:
                #print('WRITING SEQ...')
                batch_size = trg.shape[1]
                for idx in range(batch_size):
                    trg_arr = [voacb[itrg] for itrg in trg[:,idx]]
                    trg_seq = " ".join(trg_arr)
                    #print(trg_seq)
                    trg_seqs.write(trg_seq + '\n')

                    pred_arr = [vocab[ipred] for ipred in pred[:,idx]]
                    pred_seq = " ".join(pred_arr)
                    pred_seqs.write(pred_seq+'\n')

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)
