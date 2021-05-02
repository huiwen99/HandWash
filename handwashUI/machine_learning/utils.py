from model import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import datetime

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
    arr = np.expand_dims(arr, axis=0)
    arr = torch.from_numpy(arr).float()

    output = model(arr)
    pred = output.argmax(dim=1, keepdim=True).item()
    classes = {0: 'Step 1',
               1: 'Step 2 Left',
               2: 'Step 2 Right',
               3: 'Step 3',
               4: 'Step 4 Left',
               5: 'Step 4 Right',
               6: 'Step 5 Left',
               7: 'Step 5 Right',
               8: 'Step 6 Left',
               9: 'Step 6 Right',
               10: 'Step 7 Left',
               11: 'Step 7 Right'}
    prediction = classes[pred]
    return prediction

def load_model(model, model_path):
    """
    Load model from file path
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
    