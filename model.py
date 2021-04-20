import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    """"
    Adapted from: https://github.com/IDKiro/action-recognition/blob/master/model.py
    """
    def __init__(self, original_model, arch, num_classes, lstm_layers, hidden_size, fc_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size

        # select a base model
        if arch.startswith('alexnet'):
            self.features = original_model.features
            for i, param in enumerate(self.features.parameters()):
                param.requires_grad = False
            self.fc_pre = nn.Sequential(nn.Linear(9216, fc_size), nn.Dropout())
            self.rnn = nn.LSTM(input_size = fc_size,
                        hidden_size = hidden_size,
                        num_layers = lstm_layers,
                        batch_first = True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.modelName = 'alexnet_lstm'

        elif arch.startswith('resnet18'):
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            for i, param in enumerate(self.features.parameters()):
                param.requires_grad = False
            self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())
            self.rnn = nn.LSTM(input_size = fc_size,
                        hidden_size = hidden_size,
                        num_layers = lstm_layers,
                        batch_first = True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.modelName = 'resnet18_lstm'

        elif arch.startswith('resnet34'):
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            for i, param in enumerate(self.features.parameters()):
                param.requires_grad = False
            self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())
            self.rnn = nn.LSTM(input_size = fc_size,
                        hidden_size = hidden_size,
                        num_layers = lstm_layers,
                        batch_first = True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.modelName = 'resnet34_lstm'

        elif arch.startswith('resnet50'):
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            for i, param in enumerate(self.features.parameters()):
                param.requires_grad = False
            self.fc_pre = nn.Sequential(nn.Linear(2048, fc_size), nn.Dropout())
            self.rnn = nn.LSTM(input_size = fc_size,
                        hidden_size = hidden_size,
                        num_layers = lstm_layers,
                        batch_first = True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.modelName = 'resnet50_lstm'

        else:
            raise Exception("This architecture has not been supported yet")

    def init_hidden(self, num_layers, batch_size):
        return (torch.zeros(num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(num_layers, batch_size, self.hidden_size).cuda())

    def forward(self, inputs, hidden=None, steps=0):
        length = len(inputs)
        fs = torch.zeros(inputs[0].size(0), length, self.rnn.input_size).cuda()

        for i in range(length):
            f = self.features(inputs[i])
            f = f.view(f.size(0), -1)
            f = self.fc_pre(f)
            fs[:, i, :] = f

        outputs, hidden = self.rnn(fs, hidden)
        outputs = self.fc(outputs)
        return outputs


class CNN_LSTM1(nn.Module):
    """
    Model with Convolutional LSTM architecture
    """

    def __init__(self):
        super(CNN_LSTM1, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.fc = nn.Linear(64*64*16,128)
        self.rnn = nn.LSTM(128, 64, num_layers=1)
        self.classifier = nn.Linear(64, 12)

    def forward(self, inputs, hidden=None):

        # seq_len, batch_size, no. of features
        lstm_in = torch.zeros(len(inputs[0]), len(inputs), self.rnn.input_size).cuda()
        for j in range(len(inputs[0])):
            frame_batch = inputs[:,j,:,:]
            x = self.conv(frame_batch)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            lstm_in[j] = x

        outputs, hidden = self.rnn(lstm_in, hidden)
        # take the last output of sequence
        outputs = outputs[-1]
        outputs = self.classifier(outputs)
        output = F.log_softmax(outputs, dim=1)
        return output



def build_model():
    """
    Initializes and returns model
    """
    model = CNN_LSTM1() # can change this
    return model