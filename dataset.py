from utils import video_to_3d
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import PIL
import numpy as np
import os, os.path
import random
import scipy.ndimage
from skimage.util import random_noise

class Handwash_Dataset(Dataset):
    """
    Dataset Class
    """

    def __init__(self, group, output_dir='./dataset', frame_size=(128, 128), num_frames=16, data_aug=None, aug_prob = 1):
        """
        Constructor for Dataset class

        Parameters:
            - group should be set to test, train or val

        """

        # Number of frames per video
        self.num_frames = num_frames

        # Size of frame
        self.frame_size = frame_size

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
        
        self.data_aug = data_aug
        
        self.aug_prob = aug_prob
        
        # Actual Dataset - not including our own
        self.db_dir = './HandWashDataset'
        
        # Numpy Dataset
        self.output_dir = output_dir
        
        if not self.check_processed():
            print('Preprocessing of dataset, this will take long.')
            self.process_all_videos()
        
        group_dir = output_dir + '/{}'.format(self.group)
        self.dataset_paths = {'step_1': group_dir+'/Step_1', \
                              'step_2_left': group_dir+'/Step_2_Left', \
                              'step_2_right': group_dir+'/Step_2_Right', \
                              'step_3': group_dir+'/Step_3', \
                              'step_4_left': group_dir+'/Step_4_Left', \
                              'step_4_right': group_dir+'/Step_4_Right', \
                              'step_5_left': group_dir+'/Step_5_Left', \
                              'step_5_right': group_dir+'/Step_5_Right', \
                              'step_6_left': group_dir+'/Step_6_Left', \
                              'step_6_right': group_dir+'/Step_6_Right', \
                              'step_7_left': group_dir+'/Step_7_Left', \
                              'step_7_right': group_dir+'/Step_7_Right'}
        
        # Number of videos for each class in the dataset
        self.dataset_numbers = {}
        # List of filenames of videos for each class in the dataset
        self.dataset_filenames = {}
        for key in self.dataset_paths:
            path = self.dataset_paths[key]
            _, _, files = next(os.walk(path))
            self.dataset_numbers[key] = len(files)
            self.dataset_filenames[key] = files

    def check_processed(self):
        if not os.path.exists(self.output_dir):
            return False
        else:
            return True
    
    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        msg = "This is the {} dataset from the HandWash Dataset ".format(self.group)
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
        
        # sample the right number of frames from the npy arrays
        arr = np.load(filename)
        
        arr = self.time_crop(arr, self.num_frames)
        
#         frames_idxs = sorted(random.sample(range(0, len(arr)), self.num_frames))
#         arr = np.array([arr[i] for i in frames_idxs])

        if self.group == 'train' and self.data_aug is not None:
            arr = self.data_augment(arr)
            
        # resize
        arr = self.resize(arr) 
        
        # normalize
        arr = arr / 255
        arr = arr.transpose(0, 3, 1, 2)

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
        arr = arr.transpose(0, 2, 3, 1)

        # display
        for i in range(arr.shape[0]):
            frame = arr[i]
            # convert to rgb
            frame = frame[:,:,::-1]
            
            fig, ax = plt.subplots()
            ax.imshow(frame)
            ax.set_title('Frame {}'.format(i))
            ax.axis('off')
        plt.show()

    def process_all_videos(self):
        if not os.path.exists(self.db_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from https://www.kaggle.com/realtimear/hand-wash-dataset')
            
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        for group in ['train', 'val', 'test']:
            group_dir = os.path.join(self.db_dir, group)
            os.mkdir(os.path.join(self.output_dir, group))
            for step in os.listdir(group_dir):
                group_step_dir = os.path.join(group_dir, step)
                os.mkdir(os.path.join(self.output_dir, group, step))
                print("Converting to numpy files for {} ...".format(group_step_dir))
                for idx, filename in enumerate(os.listdir(group_step_dir)):
                    filepath = os.path.join(group_step_dir, filename)
                    save_dir = os.path.join(self.output_dir, group, step, '000{}.npy'.format(idx))
                    arr = video_to_3d(filepath)
                    np.save(save_dir, arr)

    def resize(self, imgs):
        resized_imgs = []
        for img in imgs:
            resized = cv2.resize(img, self.frame_size)
            resized_imgs.append(resized)
            
        return np.array(resized_imgs)
    
    def time_crop(self, buffer, clip_len):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Crop and jitter the video using indexing
        # The temporal jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,:,:,:]

        return buffer
    
    def contrast(self, imgs, factor):
        """
        Adds contrast to the sequence of images
        """
        factor = float(factor)
        arr = []
        
        for img in imgs:
            if random.randrange(0, 100) < self.aug_prob * 100:
                arr.append(np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8))
            else:
                arr.append(img)
        return np.array(arr)

    def translate(self, imgs, factor):
        """
        Do translation to the sequence of images
        """
        arr = []
        if random.randrange(0, 100) < 50: #0.5 chance shifting to the left       
            factor = -factor
        for img in imgs:
            if random.randrange(0, 100) < self.aug_prob * 100:
                trans_width = img.shape[1]*factor
                T = np.float32([[1, 0, trans_width], [0, 1, 0]])
                arr_translation = cv2.warpAffine(img, T, (img.shape[1], img.shape[0])) # width , height
                arr.append(arr_translation)
            else:
                arr.append(img)
        return np.array(arr)

    def data_augment(self, imgs):
        """
        Performs transformation on the sequence of images
        """
        arr = imgs
        if self.data_aug == "contrast":
            contrast_factor = random.uniform(0.9,1.1)
            arr = self.contrast(imgs,contrast_factor)
        elif self.data_aug == "translate":
            translate_factor = random.uniform(0.05,0.2)
            arr = self.translate(imgs, translate_factor)
            
        return arr

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
        vid = torch.from_numpy(vid).float()
        return vid, label
