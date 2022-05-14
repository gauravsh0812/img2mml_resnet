import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import random
import torchvision

class Encoder(nn.Module):

    def __init__(self, input_channel, hid_dim, n_layers, dropout, device):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)
        self.hid_dim = hid_dim
        self.conv_layer1 = nn.Conv2d(input_channel, 64, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer4 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer5 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.conv_layer6 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding =(1,1))
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.emb = nn.Embedding(256, 256)
        self.lstm = nn.LSTM(256, hid_dim, num_layers=1, dropout=0.3, bidirectional=True, batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # img = [batch, Cin, W, H]
        batch = src.shape[0]
        C_in = src.shape[1]

        # Initialize LSTM state
        hidden, cell = self.init_hidden_state(encoder_out)  # (batch_size, hid_dim)

        # src = [batch, Cin, w, h]
        # layer 1
        src = self.conv_layer1(src)
        src = F.relu(src)
        src = self.maxpool(src)
        # layer 2
        src = self.maxpool(F.relu(self.batch_norm1(self.conv_layer2(src))))
        # layer 3
        src = F.relu(self.batch_norm2(self.conv_layer3(src)))
        # layer 4
        src = self.maxpool1(F.relu(self.conv_layer4(src)))     # [B, 256, w, h]
        # layer 5
        src = F.relu(self.batch_norm3(self.conv_layer5(src)))
        # layer 6
        enc_output = F.relu(self.conv_layer6(src))    # [B, 512, w, h]

        # flatten the last two dimensions of enc_output i.e.
        # [batch, 512, W'xH']
        all_outputs = []
        for ROW in range(0, enc_output.shape[2]):
            # row => [batch, 512, W] since for each row,
            # it becomes a 2d matrix of [512, W] for all batches
            row = enc_output[:,:,ROW,:]
            row = row.permute(2,0,1)  # [W, batch, 512(enc_output)]
            pos_vec = torch.Tensor(row.shape[1]).long().fill_(ROW).to(self.device) # [batch]
            # self.emb(pos) ==> [batch, 512]
            lstm_input = torch.cat((self.emb(pos_vec).unsqueeze(0), row), dim = 0) # [W+1, batch, 512]
            lstm_output, (hidden, cell) = self.lstm(lstm_input)
            # output = [W+1, batch, hid_dimx2]
            # hidden/cell = [2x1, batch, hid_dim]
            # we want the fwd and bckwd directional final layer

            all_outputs.append(lstm_output.unsqueeze(0))

        final_encoder_output = torch.cat(all_outputs, dim =0)  #[H, W+1, BATCH, hid_dimx2]
        # modifying it to [H*W+1, batch, hid_dimx2]
        final_encoder_output = final_encoder_output.view(
                                            final_encoder_output.shape[0]*final_encoder_output.shape[1],
                                            final_encoder_output.shape[2], final_encoder_output.shape[3])

        for idx, LAYER in enumerate([hidden, cell]):
           fwd_layer, bwd_layer = torch.tanh(LAYER[0,:,:]), torch.tanh(LAYER[1,:,:])  # [batch, hid]
           if idx==0: hidden = LAYER.unsqueeze(0)     # [1, B, Hid]
           else: cell = LAYER.unsqueeze(0)

        return final_encoder_output, hidden, cell


class Attention(nn.Module):
    """
    Attention
    """

    def __init__(self, encoder_dim, hid_dim, attention_dim):
        super(Attention, self).__init__()

        self.enclayer = nn.Linear(encoder_dim, attention_dim)
        self.hidlayer = nn.Linear(hid_dim, attention_dim)
        self.enc_hidlayer = nn.Linear(hid_dim, encoder_dim)
        self.attnlayer = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.enc_1_layer = nn.Linear(encoder_dim, 1)


    def forward(self, encoder_out, hidden):
        attn1 = self.enclayer(encoder_out)   # [H*W+1, B, attention_dim]
        attn2 = self.hidlayer(hidden)       # [1, B, attn_dim]
        # print('attn1: ', attn1.shape)
        # print('attn2: ', attn2.permute(1,0,2).shape)
        net_attn = self.relu(attn1.permute(1,0,2) + attn2)   # [H*W+1, B, attn_dim]
        net_attn = self.attnlayer(net_attn)     # [H*W+1, B, 1]
        alpha = self.softmax(net_attn.permute(1,2, 0))  # [B, 1, H*W+1]
        # print('alpha:  ', alpha.shape)
        weighted_attn = torch.bmm(alpha, encoder_out).sum(dim=1) # [B,enc_dim] #(encoder_out * alpha.unsqueeze(2)).sum(dim=1)   # [B, encoder_dim]
        # print('wght_attn:  ', weighted_attn.shape)
        gate = self.sigmoid(self.enc_hidlayer(hidden.squeeze(0)))    # [B, enc_dim]
        # print('gate:  ', gate.shape)
        final_attn_encoding = torch.bmm(gate.unsqueeze(2), weighted_attn.unsqueeze(1))   # [B, enc_dim, enc_dim]
        final_attn_encoding = self.enc_1_layer(final_attn_encoding)   # [B, enc_dim, 1]
        # print('final_attn_encoding:  ', final_attn_encoding.shape)

        return final_attn_encoding.permute(2, 0, 1)    # [1, B, enc_dim]


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed_dim, encoder_dim, hid_dim, attention_dim, output_dim, n_layers, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param hid_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.attention = OpennmtAttention(encoder_dim, hid_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(output_dim, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm_input_layer = nn.Linear(embed_dim + encoder_dim, embed_dim)
        self.decode_step = nn.LSTM(embed_dim, hid_dim, num_layers=n_layers, dropout=dropout, bias=True)  # decoding LSTMCell
        # self.init_h = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(hid_dim, output_dim)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune


    def forward(self, dec_src, encoder_out, hidden, cell):
        # Embedding
        # print('dec_src:  ', dec_src)
        embeddings = self.embedding(dec_src.int().unsqueeze(0))  # (1, batch_size, embed_dim)
        # print('dec emb shape: ', embeddings.shape)
        # print('h shape:  ', hidden.shape)
        # hidden shape = [1, B, hid]

        # Calculate attention
        final_attn_encoding = self.attention(encoder_out, hidden)    # [ 1, B, enc-dim]

        # lstm input
        # print(embeddings.shape)
        # print(final_attn_encoding.shape)
        lstm_input = torch.cat((embeddings, final_attn_encoding), dim=2)    # [1, B, enc+embed]
        lstm_input = self.lstm_input_layer(lstm_input)                      # [1, B, embed]
        lstm_output, (hidden, cell) = self.decode_step(lstm_input, (hidden, cell))    # H: [1, B, hid]     O: [1, B, Hid*2]
        predictions = self.fc(lstm_output)  # [1, Batch, output_dim]

        return predictions.squeeze(0), hidden, cell


class Img2Seq(nn.Module):
    """
    Calling class
    """
    def __init__(self, encoder, decoder, device, encoder_dim, hid_dim):
        super(Img2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.init_h = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial cell state of LSTMCell

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        # print('encoder_out:  ', encoder_out.shape)
        h = self.init_h(mean_encoder_out)  # (batch_size, hid_dim)
        c = self.init_c(mean_encoder_out)
        return h.unsqueeze(0), c.unsqueeze(0)

    def forward(self, src, trg,  vocab, write_flag=False, teacher_force_flag=False, teacher_forcing_ratio=0):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_dim = self.decoder.output_dim

        # to store all separate outputs of individual token
        outputs = torch.zeros(trg_len, batch_size, trg_dim).to(self.device) #[trg_len, batch, output_dim]
        # for each token, [batch, output_dim]

        # run the encoder --> get flattened FV of images
        encoder_out, hidden, cell = self.encoder(src)       # enc_output: [HxW+1, B. H*2]   Hid/cell: [1, B, Hid]

        dec_src = trg[0,:]   # [1, B]

        if write_flag:
            pred_seq_per_batch = torch.zeros(trg.shape)
            init_idx = vocab.stoi['<sos>']  # 2
            pred_seq_per_batch[0,:] = torch.full(dec_src.shape, init_idx)

        for t in range(1, trg_len):

            output, (hidden, cell) = self.decoder(dec_src, encoder_out, hidden, cell)     # O: [B, out]   H: [1, B, Hid]
            outputs[t]=output
            top1 = output.argmax(1)     # [batch_size]

            if write_flag:
                pred_seq_per_batch[t,:] = top1
            # decide if teacher forcing shuuld be used or not
            teacher_force = False
            if teacher_force_flag:
                teacher_force = random.random() < teacher_forcing_ratio

            dec_src = trg[t] if teacher_force else top1


        if  write_flag: return outputs, pred_seq_per_batch, self.encoder, self.decoder
        else: return outputs, self.encoder, self.decoder
