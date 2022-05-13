# -*- coding: utf-8 -*-

import torch
# from preprocessing.preprocess_images import preprocess_images

def evaluate(model, vocab, batch_size, test_dataloader, criterion, device, write_file):

    model.eval()

    epoch_loss = 0

    trg_seqs = open('logs/test_targets.txt', 'w')
    pred_seqs = open('logs/test_predicted.txt', 'w')

    with torch.no_grad():

        for i, (img, mml) in enumerate(test_dataloader):

            if i%50==0: print(f'test_{i}')
            # initailize the hidden state
            trg = mml.to(device, dtype=torch.int64)
            # print('@test trg shape:  ', trg.shape)
            batch_size = trg.shape[1]

            # loading image Tensors
            srcTensor = []
            for _i in img:
                srcTensor.append(torch.load(f'data/image_tensors/{_i}.txt'))
            src = torch.stack(srcTensor).to(device)


            # trg = batch.mml.to(device)

            output, pred, encoder, decoder = model(src, trg, vocab, True)   # turn off teacher_forcing

            # translating and storing trg and pred sequences in batches
            if write_file:
                #print('WRITING SEQ...')
                batch_size = trg.shape[1]
                for idx in range(batch_size):

                    trg_arr = [vocab.itos[itrg] for itrg in trg[:,idx]]
                    trg_seq = " ".join(trg_arr)
                    # print(trg_seq)
                    trg_seqs.write(trg_seq + '\n')

                    pred_arr = [vocab.itos[ipred] for ipred in pred.int()[:,idx]]
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
            net_loss = epoch_loss / len(test_dataloader)

        return net_loss, encoder, decoder
