import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class DCAE(nn.Module):
    def __init__(self, channel=3, height=224, width=224, z_dim=50):
        super(DCAE, self).__init__()
        # Encoder
        layer0=8
        layer1=32
        layer2=64
        layer3=128
        layer4=256
        linear_dim=layer4*14*14
        self.e_conv1 = nn.Conv2d(channel, layer1, 3, 2,1) #b,3,i,i -> #b,32,i,i  #in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1
        self.e_conv2 = nn.Conv2d(layer1, layer2, 3, 2,1) #b,64,i,i
        self.e_conv3 = nn.Conv2d(layer2, layer3, 3, 2,1) #b,128,i,i
        self.e_conv4 = nn.Conv2d(layer3, layer4, 3, 2,1) #b,layer4,i,i
        self.e_fl = nn.Flatten()
        self.e_fc1 = nn.Linear(linear_dim,1000) 
        self.e_bn1 = nn.BatchNorm1d(1000)
        self.e_fc2 = nn.Linear(1000,z_dim)
        self.e_bn2 = nn.BatchNorm1d(z_dim)
        # Decoder
        self.d_fc1 = nn.Linear(z_dim,1000)
        self.d_bn1 = nn.BatchNorm1d(1000)
        self.d_fc2 = nn.Linear(1000,linear_dim)
        self.d_bn2 = nn.BatchNorm1d(linear_dim)
        self.d_rs = Reshape(-1, layer4, 14, 14)
        self.d_dconv1 = nn.ConvTranspose2d(layer4, layer3, 4, 2,1)
        self.d_dconv2 = nn.ConvTranspose2d(layer3, layer2, 4, 2,1)
        self.d_dconv3 = nn.ConvTranspose2d(layer2, layer1, 4, 2,1)
        self.d_dconv4 = nn.ConvTranspose2d(layer1, channel, 4, 2,1)
        print("DCAE init")
        return

    def _encoder(self,x):
        x = F.relu(self.e_conv1(x))
        x = F.relu(self.e_conv2(x))
        x = F.relu(self.e_conv3(x))
        x = F.relu(self.e_conv4(x))
        x = self.e_fl(x)
        x = self.e_bn1(self.e_fc1(x))
        x = self.e_bn2(self.e_fc2(x))
        return x

    def _decoder(self,x):
        x = self.d_bn1(self.d_fc1(x))
        x = self.d_bn2(self.d_fc2(x))
        x = self.d_rs(x)
        x = F.relu(self.d_dconv1(x))
        x = F.relu(self.d_dconv2(x))
        x = F.relu(self.d_dconv3(x))
        x = nn.Sigmoid()(self.d_dconv4(x))
        return x

    def forward(self,x):
        z = self._encoder(x)
        x = self._decoder(z)
        return x,z

class LSTM(nn.Module):
    def __init__(self, input_size=3, output_size=1, hidden_size=10,batch_first=True):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=batch_first)
        self.output_layer = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        print("LSTM init")
        return

    def forward(self,inputs):
        lstm_out, state = self.rnn(inputs)
        output = self.output_layer(lstm_out)
        return output, state

if __name__ == '__main__':
    dcae = DCAE()
    print(dcae)
    lstm = LSTM()
    print(lstm)
