import torch
from torch import nn 
import torch.nn.functional as F

class Conv_GAIN_G(nn.Module):
    def __init__(self, kn1, kn2, hidden_channel, pooling_size, output_size):
        super(Conv_GAIN_G, self).__init__()
        self.conv1 = nn.Conv2d(2, hidden_channel, kernel_size=kn1)
        self.pooling = nn.MaxPool2d(pooling_size)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel*2, kernel_size=kn2)
        self.out = nn.Linear(hidden_channel*4, output_size)

    def forward(self, X):
        elm_n = X.shape[0]
        out = torch.relu(self.conv1(X))
        out = self.pooling(out)
        out = torch.relu(self.conv2(out))
        out = self.pooling(out)
        out = out.reshape(elm_n, -1)
        out = self.out(out)
        return out

class Conv_GAIN_D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Conv_GAIN_D, self).__init__()
        self.input = nn.Linear(input_dim*2, input_dim)
        self.hidden = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x, h):
        inputs = torch.cat((x,h), 1)
        G_h1 = torch.relu(self.input(inputs))
        G_h2 = torch.relu(self.hidden(G_h1))
        G_prob = self.output(G_h2)
        return G_prob

