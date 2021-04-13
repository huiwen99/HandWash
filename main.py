from video_processing import *

if __name__ == "__main__":
    num_classes = 12
    file_path = 'HandWashDataset/Step_1/HandWash_001_A_01_G01.mp4'
    test_frames = video_to_3d(file_path, 10)
    print(test_frames)