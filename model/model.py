import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

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

class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, embed_dim, hid_dim, output_dim, n_layers, encoder_dim=2048, dropout=0.5):
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
        # self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout

        # self.attention = Attention(encoder_dim, hid_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(output_dim, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, hid_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, hid_dim)  # linear layer to find initial cell state of LSTMCell
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


    def forward(self, dec_src, hidden, cell):
        # Embedding
        embeddings = self.embedding(dec_src.unsqueeze(0))  # (1, batch_size, embed_dim)
        print('dec emb shape: ', embeddings.shape)
        print('h shape:  ', hidden.shape)
        lstm_output, (hidden, cell) = self.decode_step(embeddings, (hidden, cell))
        prediction = self.fc(lstm_output)  # [1, Batch, output_dim]

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
        h = self.init_h(mean_encoder_out)  # (batch_size, hid_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, trg_field, src, trg,  write_flag=False, teacher_force_flag=False, teacher_forcing_ratio=0):

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

        dec_src = trg[0,:]   # [1, B]

        if write_flag:
            pred_seq_per_batch = torch.zeros(trg.shape)
            init_idx = trg_field.vocab.stoi[trg_field.init_token] #trg_vocab_stoi[trg_field.init_token]
            pred_seq_per_batch[0,:] = torch.full(dec_src.shape, init_idx)

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(dec_src, hidden.unsqueeze(0), cell.unsqueeze(0))
            outputs[t]=output
            top1 = output.argmax(1)     # [batch_size]

            if write_flag:
                pred_seq_per_batch[t,:] = top1
            # decide if teacher forcing shuuld be used or not
            teacher_force = False
            if teacher_force_flag:
                teacher_force = random.random() < teacher_forcing_ratio

            dec_src = trg[t] if teacher_force else top1


        if  write_flag: return outputs, pred_seq_per_batch
        else: return outputs
