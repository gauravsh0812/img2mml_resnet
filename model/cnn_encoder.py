import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import  random

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
        self.AdapAvgPool = nn.AdaptiveAvgPool2d(79,5)
        self.AdapMaxPool = nn.AdaptiveMaxPool2d(79,5)
        self.emb = nn.Embedding(256, 256)
        self.lstm = nn.LSTM(256, hid_dim, num_layers=1, dropout=0.3, bidirectional=True)
        self.fc_wh = nn.Linear(256*79*11, 2)
        self.fc_hidden = nn.Linear(2*256, self.hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # img = [batch, Cin, W, H]
        batch = src.shape[0]
        C_in = src.shape[1]

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
        # src = F.relu(self.batch_norm3(self.conv_layer5(src)))
        # layer 6
        # src = F.relu(self.conv_layer6(src))    # [B, 512, w, h]
        # print('shape after layer 6:  ', src.shape)

        enc1 = self.AdapAvgPool(src)           # [B, C, 79, 11]
        enc2 = self.AdapMaxPool(src)           # [B, C, 79, 11]
        enc = torch.cat((enc1, enc2), dim=1)   # [B, 2*C, 79, 11]
        enc_output = torch.flatten(enc, start_dim=-2, end_dim=-1)  # [B, 2C, W'xH']
        enc_output = self.fc_wh(enc_output).permute(0, 2, 1) # [B, 2, 2C]
        enc_output = self.fc_hidden(enc_output).permute(1, 0, 2)   # [2, B, hid]

        return enc_output
