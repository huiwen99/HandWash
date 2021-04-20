from model import *
import datetime
from video_processing import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(model, device, train_loader, val_loader, optimizer, epoch):
    """
    Trains the model on training data
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # convert one-hot to numerical categories
        target = torch.argmax(target, dim=1).long()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(model, device, train_loader)
    print('Train Epoch: {} @ {} \nTrain Loss: {:.4f} - Train Accuracy: {:.1f}%'.format(
        epoch, datetime.datetime.now().time(), train_loss, train_acc))

    val_loss, val_acc = evaluate(model, device, val_loader)
    print("Val Loss: {:.4f} - Val Accuracy: {:.1f}%".format(val_loss, val_acc))

    return train_loss, train_acc, val_loss, val_acc


def evaluate(model, device, data_loader):
    """
    Evaluates the model and returns loss, accuracy
    """
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1).long()
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)

    return loss, acc

def predict(model, video_path, num_frames):
    """
    Predicts the label of the video given its filepath
    """
    frame_size = (64,64)
    arr = video_to_3d(video_path, num_frames, frame_size, color=True, random_frames=False)
    arr = arr.transpose(0, 3, 1, 2)
    arr = np.asarray(arr) / 255
    arr = torch.from_numpy(arr).float()
    output = model(arr)
    pred = output.argmax(dim=1, keepdim=True).item()
    classes = {0: 'step_1',
               1: 'step_2_left',
               2: 'step_2_right',
               3: 'step_3',
               4: 'step_4_left',
               5: 'step_4_right',
               6: 'step_5_left',
               7: 'step_5_right',
               8: 'step_6_left',
               9: 'step_6_right',
               10: 'step_7_left',
               11: 'step_7_right'}
    prediction = classes[pred]
    return prediction

def load_model(model_path):
    """
    Load model from file path
    """
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    return model

def plot_curves(train_arrs, val_arrs, plot_name):
    """
    Plots training and testing learning curves over successive epochs
    """
    plt.plot(train_arrs, label="Train")
    plt.plot(val_arrs, label="Val")
    plt.title(plot_name)
    plt.legend()
    plt.show()