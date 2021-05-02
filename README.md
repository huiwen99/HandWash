# HandWash

## Dataset
Download the pre-processed numpy dataset to the root directory: `wget https://storage.googleapis.com/dl-big-project/dataset.zip`  
Unzip the file: `unzip ./dataset.zip`

## Reproducibility
Our model weights are too large to fit in this repository. Download the weights of our best model in root directory by running the command:  
`wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt`

## Instructions to run python files in the notebook
1. `train.py`: Trains the chosen architecture on the numpy dataset. The default model is CNN-LSTM with custom CNN layers.  

    To train the default model, run the following command:
    `!python train.py`  

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

     After training the model and saving it, we can evaluate the model on test set.  
     For example, the model is saved as `"./alexnet_128.pt"`.  
     
     Run the following command:  
     `!python evaluate.py --model_dir "./alexnet_128.pt" --arch alexnet`  
     where the `arch` parameter has to match the architecture of the saved model.

    Optional parameters:  
    `--confusionMatrix`: print confusion matrix if set to True  
    `--cuda`: enable cuda   

3. `predict.py`: Predicts the class of a video using the trained model.  

    After training the model and saving it, we can use the model to predict the class of handwash videos.

    Run the following command:  
    `%run predict.py --checkpoint "./alexnet_128.pt" --video_path video_file_path`  
    where `video_file_path` is the file path to the video. 
