# HandWash

## Dataset
Download the pre-processed numpy dataset to the root directory: `wget https://storage.googleapis.com/dl-big-project/dataset.zip`

## Instructions to run python files
1. `train.py`: Trains the chosen architecture on the numpy dataset. The default model is CNN-LSTM with custom CNN layers.  

    To train the default model, run the following command:
    `python train.py`

    Optional parameters:  
    `--arch`: set architecture (either `alexnet` or `resnet50`)      
    `--epochs`: set number of training epochs  
    `--batch`: set batch size  
    `--num_frames`: set number of frames per video
    `--lr`: set learning rate  
    `--beta1`: set first momentum term for Adam optimizer  
    `--beta2`: set second momentum term for Adam optimizer  
    `--weight_decay`: set weight decay for regularization on loss function  
    `--gamma`: set gamma for learning rate scheduler  
    `--step_size`: set step size for learning rate scheduler  
    `--cuda`: enable cuda training  
    `--checkpoint`: filepath to a checkpoint to load model  
    `--save_dir`: filepath to save the model  
    `--data_aug`: enable data augmentation during training  

2. `evaluate.py`: Evaluate the trained model on test set.  

     After training the model and saving it to the `./save_weights` directory, we can evaluate the model on test set.  
     For example, the model is saved as `"./save_weights/alexnet_aug.pt"`.  
     
     Run the following command:  
     `python evaluate.py`   


3. `predict.py`: Predicts the class of a video using the trained model.  

    After training the model and saving it to the `./save_weights` directory, we can use the model to predict the class of handwash videos.

    Run the following command:  
    `python predict.py --arch alexnet --checkpoint "./save_weights/alexnet_aug.pt" --video_path video_file_path`  
    where `video_file_path` is the file path to the video. 
