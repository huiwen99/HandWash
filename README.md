# HandWash

## Dataset
Download the pre-processed numpy dataset to the root directory: `wget https://storage.googleapis.com/dl-big-project/dataset.zip`

## Instructions to run python files
1. `train.py`: Trains the chosen architecture on the numpy dataset. The default model is CNN-LSTM with custom CNN layers.  

    To train the default model, run the following command:
    `python train.py`
    
    To load in the best model: `wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt`

    Optional parameters:  
    `--arch`: set architecture (either `convlstm` or `alexnet` or `resnet50` or `custom`)      
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
    `--data_aug`: set data augmentation type: None or `constrast` or `translate`   
    `--aug_prob`: decide on the probability of the dataset to perform data augmentation

2. `evaluate.py`: Evaluate the trained model on test set.  

     After training the model and saving it to the `./save_weights` directory, we can evaluate the model on test set.  
     For example, the model is saved as `"./save_weights/alexnet_aug.pt"`.  
     
     Run the following command:  
     `python evaluate.py`   

    Optional parameters:  
    `--arch`: set architecture (either `convlstm` or `alexnet` or `resnet50` or `custom`)        
    `--dataset`: choose the dataset to evaluate on (either `validation` or `test`)  
    `--batch`: set batch size (for evaluation on validation set)  
    `--model_dir`: filepath to the saved model   
    `--confusionMatrix`: print confusion matrix if set to True  
    `--cuda`: enable cuda 

3. `predict.py`: Predicts the class of a video using the trained model.  

    After training the model and saving it to the `./save_weights` directory, we can use the model to predict the class of handwash videos.

    Run the following command:  
    `python predict.py --arch alexnet --checkpoint "./save_weights/alexnet_aug.pt" --video_path video_file_path`  
    where `video_file_path` is the file path to the video. 
