# -*- coding: utf-8 -*-

import torch
from preprocessing.preprocess_images import preprocess_images

def evaluate(trg_field, model, batch_size, iterator, criterion, device, write_file):

    model.eval()

    epoch_loss = 0

    trg_seqs = open('logs/test_targets.txt', 'w')
    pred_seqs = open('logs/test_predicted.txt', 'w')

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            # if i%50==0:print(f'test-{i}')
            trg = batch.mml.to(device)
            batch_size = trg.shape[1]
            # h = model.encoder.init_hidden(batch_size)

            img_names = batch.id
            src = preprocess_images(img_names, 'data/images/')
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
                    trg_arr = [trg_field.vocab.itos[itrg] for itrg in trg[:,idx]]#[trg_vocab_itos[itrg] for itrg in trg[:,idx]]
                    trg_seq = " ".join(trg_arr)
                    #print(trg_seq)
                    trg_seqs.write(trg_seq + '\n')

                    pred_arr = [trg_field.vocab.itos[ipred] for ipred in pred[:,idx].int()]#[trg_vocab_itos[ipred] for ipred in pred[:,idx].int()]
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