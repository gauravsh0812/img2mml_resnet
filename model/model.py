import torch
import random
from torch import nn
import torchvision

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet18(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

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
        attn1 = self.enclayer(encoder_out)   # [B, num_pixels, attention_dim]
        attn2 = self.hidlayer(hidden)       # [1, B, attn_dim]
        # print('attn1: ', attn1.shape)
        # print('attn2: ', attn2.permute(1,0,2).shape)
        net_attn = self.relu(attn1.permute(1,0,2) + attn2)   # [num, B, attn_dim]
        # print('net attn: ', net_attn.shape)
        #net_attn = self.attnlayer(net_attn).squeeze(2)     # [num, B]
        net_attn = self.attnlayer(net_attn)     # [num, B, 1]
        alpha = self.softmax(net_attn.permute(1,2, 0))  # [B, 1, num]
        # print('alpha:  ', alpha.shape)
        weighted_attn = torch.bmm(alpha, encoder_out).sum(dim=1) # [B,enc_dim] #(encoder_out * alpha.unsqueeze(2)).sum(dim=1)   # [B, encoder_dim]
        # print('wght_attn:  ', weighted_attn.shape)
        # gate = self.sigmoid(self.enc_hidlayer(hidden.squeeze(0)))    # [B, enc_dim]
        # print('gate:  ', gate.shape)
        # final_attn_encoding = gate * weighted_attn   # [B, enc_dim]
        # final_attn_encoding = torch.bmm(gate.unsqueeze(2), weighted_attn)   # [B, enc_dim, enc_dim]
        # final_attn_encoding = self.enc_1_layer(final_attn_encoding)   # [B, enc_dim, 1]
        # print('final_attn_encoding:  ', final_attn_encoding.shape)

        # return final_attn_encoding.permute(2, 0, 1)#.unsqueeze(0)
        return weighted_attn.unsqueeze(0)


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

        self.attention = Attention(encoder_dim, hid_dim, attention_dim)  # attention network

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
        lstm_input = torch.cat((embeddings, final_attn_encoding), dim=2)
        lstm_input = self.lstm_input_layer(lstm_input)
        lstm_output, (hidden, cell) = self.decode_step(lstm_input, (hidden, cell))
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
        encoder_out = self.encoder(src)       # [B, e_i, e_i, encoder_dim or C_out]
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_out.shape[-1])  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Initialize LSTM state
        hidden, cell = self.init_hidden_state(encoder_out)  # (batch_size, hid_dim)
#         print("Initialize LSTM states")

        dec_src = trg[0,:]   # [1, B]

        if write_flag:
            pred_seq_per_batch = torch.zeros(trg.shape)
            init_idx = vocab.stoi['<sos>']  # 2
            pred_seq_per_batch[0,:] = torch.full(dec_src.shape, init_idx)

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(dec_src, encoder_out, hidden, cell)
#            print("ran decoder")
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
