from video_processing import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os, os.path
import numpy as np
import matplotlib.pyplot as plt

class Handwash_Dataset(Dataset):
    """
    Dataset Class
    """

    def __init__(self, group):
        """
        Constructor for generic Dataset class - simply assembles important parameters in attributes.

        Parameters:
            - group should be set to test, train or val

        """

        # Number of frames per video
        self.num_frames = 10

        # Size of frame
        self.frame_size = (64,64)

        # 12 classes: 1 for each of the 7 steps (hand specified if any)
        self.classes = {0: 'step_1',
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

        # Dataset should belong to only one group: train, test or val
        self.group = group

        # Path to videos in the dataset
        dir = './dataset/{}'.format(self.group)
        self.dataset_paths = {'step_1': dir+'/Step_1', \
                              'step_2_left': dir+'/Step_2_Left', \
                              'step_2_right': dir+'/Step_2_Right', \
                              'step_3': dir+'/Step_3', \
                              'step_4_left': dir+'/Step_4_Left', \
                              'step_4_right': dir+'/Step_4_Right', \
                              'step_5_left': dir+'/Step_5_Left', \
                              'step_5_right': dir+'/Step_5_Right', \
                              'step_6_left': dir+'/Step_6_Left', \
                              'step_6_right': dir+'/Step_6_Right', \
                              'step_7_left': dir+'/Step_7_Left', \
                              'step_7_right': dir+'/Step_7_Right'}

        # Number of videos for each class in the dataset
        self.dataset_numbers = {}
        # List of filenames of videos for each class in the dataset
        self.dataset_filenames = {}
        for key in self.dataset_paths:
            path = self.dataset_paths[key]
            _, _, files = next(os.walk(path))
            self.dataset_numbers[key] = len(files)
            self.dataset_filenames[key] = files



    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        msg = "This is the {} dataset from the HandWash Dataset".format(self.group)
        msg += "used for the Big Project in the 50.039 Deep Learning class. \n"
        msg += "It contains a total of {} videos. \n".format(sum(self.dataset_numbers.values()))
        msg += "The videos are stored in the following locations "
        msg += "and each one contains the following number of videos:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} videos.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_video(self, class_val, index_val):
        """
        Opens video with specified parameters.

        Parameters:
            - class_val should be set to one of the classes e.g.: 'step_1', 'step_2_left'
            - index_val should be an integer with values between 0 and the maximal number of videos in dataset.

        Returns processed video as numpy array.
        """

        # Asserts checking for consistency in passed parameters
        err_msg = "Error - class_val variable is incorrect."
        assert class_val in self.classes.values(), err_msg
        max_val = self.dataset_numbers['{}'.format(class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of videos."
        err_msg += "\n(In {}, you have {} videos.)".format(class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # open video as numpy array
        filenames = self.dataset_filenames['{}'.format(class_val)]
        filename = '{}/{}'.format(self.dataset_paths['{}'.format(class_val)], filenames[index_val])
        if self.group == 'train':
            random_frames = True
        else:
            random_frames = False
        arr = video_to_3d(filename, self.num_frames, self.frame_size, color=True, random_frames=random_frames)

        # normalize
        arr = np.asarray(arr) / 255

        return arr

    def show_video(self, class_val, index_val):
        """
        Opens, then displays video frames with specified parameters

        Parameters:
            - class_val should be set to one of the classes e.g.: 'step_1', 'step_2_left'
            - index_val should be an integer with values between 0 and the maximal number of videos in dataset.
        """

        # open video
        arr = self.open_video(class_val, index_val)

        # display
        for i in range(arr.shape[0]):
            frame = arr[i]
            fig, ax = plt.subplots()
            ax.imshow(frame)
            ax.set_title('Frame {}'.format(i))
            ax.axis('off')
        plt.show()

    def __len__(self):
        """
        Length special method, returns the number of videos in dataset.
        """

        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the video and its label as a one hot vector, both
        in torch tensor format in dataset.
        """

        # Get item special method
        one_hot = np.zeros(len(self.classes))

        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1]) + first_val
        third_val = int(list(self.dataset_numbers.values())[2]) + second_val
        fourth_val = int(list(self.dataset_numbers.values())[3]) + third_val
        fifth_val = int(list(self.dataset_numbers.values())[4]) + fourth_val
        sixth_val = int(list(self.dataset_numbers.values())[5]) + fifth_val
        seventh_val = int(list(self.dataset_numbers.values())[6]) + sixth_val
        eighth_val = int(list(self.dataset_numbers.values())[7]) + seventh_val
        nineth_val = int(list(self.dataset_numbers.values())[8]) + eighth_val
        tenth_val = int(list(self.dataset_numbers.values())[9]) + nineth_val
        eleventh_val = int(list(self.dataset_numbers.values())[10]) + tenth_val

        if index < first_val:
            class_num = 0
        elif index < second_val:
            class_num = 1
            index -= first_val
        elif index < third_val:
            class_num = 2
            index -= second_val
        elif index < fourth_val:
            class_num = 3
            index -= third_val
        elif index < fifth_val:
            class_num = 4
            index -= fourth_val
        elif index < sixth_val:
            class_num = 5
            index -= fifth_val
        elif index < seventh_val:
            class_num = 6
            index -= sixth_val
        elif index < eighth_val:
            class_num = 7
            index -= seventh_val
        elif index < nineth_val:
            class_num = 8
            index -= eighth_val
        elif index < tenth_val:
            class_num = 9
            index -= nineth_val
        elif index < eleventh_val:
            class_num = 10
            index -= tenth_val
        else:
            class_num = 11
            index -= eleventh_val

        class_val = self.classes[class_num]
        one_hot[class_num] = 1
        label = torch.Tensor(one_hot)

        vid = self.open_video(class_val, index)
        vid = torch.from_numpy(vid)
        return vid, label


def dataloader(group, batch_size=4, shuffle=True):
    """
    Loads Dataset and returns DataLoader
    """
    dataset = Handwash_Dataset(group)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader