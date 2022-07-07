# -*- coding: utf-8 -*-

import torch
# from preprocessing.preprocess_images import preprocess_images

def evaluate(model, epoch, vocab, batch_size, test_dataloader, criterion, device, write_file):

    model.eval()

    # test_dataloader.sampler.set_epoch(i)

    epoch_loss = 0

    if write_file:
        trg_seqs = open('logs/test_targets_epoch_{epoch}.txt', 'w')
        pred_seqs = open('logs/test_predicted_epoch_{epoch}.txt', 'w')

    with torch.no_grad():

        for i, (img, mml) in enumerate(test_dataloader):

            # if i%50==0: print(f'test_{i}')
            # initailize the hidden state
            trg = mml
            trg = trg.permute(1,0)
            trg.to(device, dtype=torch.int64)
            # print('@test trg shape:  ', trg.shape)
            batch_size = trg.shape[0]

            # loading image Tensors
            #srcTensor = []
            #for _i in img:
            #    srcTensor.append(torch.load(f'data/image_tensors/{int(_i)}.txt'))
            #src = torch.stack(srcTensor).to(device)
            src = img
            src.to(device)

            # trg = batch.mml.to(device)

            # output, pred, encoder, decoder = model(src, trg, vocab, True)   # turn off teacher_forcing
            output, pred = model(src, trg, vocab, True)   # turn off teacher_forcing
            output = output.permute(1,0,2)

            # translating and storing trg and pred sequences in batches
            if write_file:
                #print('WRITING SEQ...')
                for idx in range(batch_size):
                    #print(trg[:,idx])
                    trg_arr = [vocab.itos[int(itrg)] for itrg in trg.int()[idx,:]]
                    trg_seq = " ".join(trg_arr)
                    # print(trg_seq)
                    trg_seqs.write(trg_seq + '\n')

                    #print(trg.shape,  pred.shape)
                    pred_arr = [vocab.itos[ipred] for ipred in pred.int()[idx,:]]
                    pred_seq = " ".join(pred_arr)
                    pred_seqs.write(pred_seq+'\n')

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            trg = trg.permute(1,0)
            # print(trg.shape,  output.shape)
            #print(' ')
            output_dim = output.shape[-1]
            output = output[1:].contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            # print('testing scripts:  ', output.shape,  trg.shape)
            #print(' ')
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg.to(torch.int64))

            epoch_loss += loss.item()
            net_loss = epoch_loss / len(test_dataloader)

        return net_loss#, encoder, decoder
