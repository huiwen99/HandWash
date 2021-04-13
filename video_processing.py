import cv2
import random

def video_to_3d(filename, target_num_frames, img_size=(64,64), color=True, random_frames=True):
    """
    Converts a video into n number of frames in numpy format
    """
    # Process the video
    cap = cv2.VideoCapture(filename)
    total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if random_frames:
        # Get indices of the frames in randomly, and then sort
        frames_idxs = random.sample(range(0, total_num_frames), target_num_frames)
        frames_idxs.sort()
    else:
        # Get indices of the frames in equal intervals
        ratio = total_num_frames / target_num_frames
        frames_idxs = [int(x * ratio) for x in range(target_num_frames)]
    
    # Return n frames for each video in numpy format
    framearray = []
    for i in range(target_num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()

        # Resize the image to feed into neural network
        frame = cv2.resize(frame, img_size)
    
        if color: # 3 channels
            framearray.append(frame)
        else: # 1 channel
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()
    
    return np.array(framearray)