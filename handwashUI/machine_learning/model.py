import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    """
    Model with Convolutional LSTM architecture
    """

    def __init__(self, arch):
        super(CNN_LSTM, self).__init__()
        
         # select a base model
        if arch.startswith('alexnet'):
            net = models.alexnet(pretrained=True)
            self.features = net.features
            for param in self.features.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(2304, 128), nn.Dropout())
        elif arch.startswith('resnet50'):
            net = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(net.children())[:-1])
            for param in self.features.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(nn.Linear(2048, 128), nn.Dropout())
        else:
            self.features = nn.Conv2d(3, 8, 3, stride=1, padding=1)
            self.fc = nn.Sequential(nn.Linear(8*64*64, 128), nn.Dropout())
            
        self.rnn = nn.LSTM(128, 2304, num_layers = 1)
        self.fc2 = nn.Sequential(nn.Linear(2304, 512), nn.ReLU(), nn.Dropout())
        self.classifier = nn.Linear(512, 12)

    def forward(self, inputs, hidden=None):
        seq_length = len(inputs[0])
        batch_size = len(inputs)
        lstm_in = torch.zeros(seq_length, batch_size, self.rnn.input_size)
            
        for j in range(seq_length):
            x = inputs[:,j,:,:]
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            lstm_in[j] = x
            
        outputs, hidden = self.rnn(lstm_in, hidden)
        # take the last output of sequence
        outputs = outputs[-1]
        outputs = self.fc2(outputs)
        outputs = self.classifier(outputs)
        output = F.log_softmax(outputs, dim=1)
        return output

def build_model(arch):
    """
    Initialize and returns model
    """

    return CNN_LSTM(arch)