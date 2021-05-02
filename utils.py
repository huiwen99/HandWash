from model import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
import random
import pandas as pd

def video_to_3d(filename, img_size=(128,128)):
    """
    Preprocess video into frames
    """
    # Process the video
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make sure the video has at least 32 frames
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 32:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 32:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 32:
                EXTRACT_FREQUENCY -= 1

    # Return n frames for each video in numpy format
    framearray = []
    count = 0
    retaining = True

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            frame = cv2.resize(frame, img_size)
            framearray.append(frame)

        if len(framearray) == 32:
            break

        count += 1

    capture.release()

    return np.array(framearray)

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

    train_loss, train_acc,train_cm = evaluate(model, device, train_loader)
    print('Train Epoch: {} @ {} \nTrain Loss: {:.4f} - Train Accuracy: {:.1f}%'.format(
        epoch, datetime.datetime.now().time(), train_loss, train_acc))

    val_loss, val_acc,val_cm = evaluate(model, device, val_loader)
    print("Val Loss: {:.4f} - Val Accuracy: {:.1f}%".format(val_loss, val_acc))

    return train_loss, train_acc, val_loss, val_acc

def evaluate(model, device, data_loader):
    """
    Evaluates the model and returns loss, accuracy and confusion matrix
    """
    confusion = {} #key: predicted_actual
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
            for ind in range(len(pred)): #for confusion matrix
                key = str(int(pred[ind]))+"_"+str(int(target[ind]))
                if key in confusion:
                    confusion[key] = confusion[key]+1
                else:
                    confusion[key] = 1
  
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    df = getConfusiondf(confusion)
    return loss, acc,df

def predict(model, video_path, num_frames = 16):
    """
    Predicts the label of the video given its filepath
    """
    arr = video_to_3d(video_path)
    
    # randomly select time index for temporal jittering
    time_index = np.random.randint(arr.shape[0] - num_frames)

    # Crop and jitter the video using indexing
    # The temporal jitter takes place via the selection of consecutive frames
    arr = arr[time_index:time_index + num_frames,:,:,:]

    # resize
    arr = np.asarray(arr) / 255
    arr = arr.transpose(0, 3, 1, 2)
    
    # display
    display_frames(arr)

    arr = np.expand_dims(arr, axis=0)
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

def display_frames(arr):
    display_arr = arr.transpose(0, 2, 3, 1)

    fig, ax = plt.subplots(nrows=4, ncols=4,figsize=(15,15))
    for i in range(display_arr.shape[0]):
        frame = display_arr[i]
        frame = frame[:,:,::-1]
        ax.ravel()[i].imshow(frame)
        ax.ravel()[i].set_title('Frame {}'.format(i))
        ax.ravel()[i].set_axis_off()
    plt.show()

def load_model(model, model_path):
    """
    Load model from file path
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
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
    
def getConfusiondf(d):
    final =[]
    for p in range(12):
        row = []
        for t in range(12):
            key = str(p)+"_"+str(t)
            if key in d:
                row.append(d[key])
            else:
                row.append(0)
        final.append(row)
    df = pd.DataFrame(final, columns=[i for i in range(12)])
    return df   

def evaluate(model, device, data_loader):
    """
    Evaluates the model and returns loss, accuracy and confusion matrix
    """
    confusion = {} #key: predicted_actual
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
            for ind in range(len(pred)): #for confusion matrix
                key = str(int(pred[ind]))+"_"+str(int(target[ind]))
                if key in confusion:
                    confusion[key] = confusion[key]+1
                else:
                    confusion[key] = 1
  
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    df = getConfusiondf(confusion)
    return loss, acc,df