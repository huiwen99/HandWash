# HandWash

## Dataset
Download the pre-processed numpy dataset to the root directory: `wget https://storage.googleapis.com/dl-big-project/dataset.zip`  
Unzip the file: `unzip ./dataset.zip`

## Reproducibility
Our model weights are too large to fit in this repository. Download the weights of our best model into the `save_weights` folder by running the command:  
`wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt`

** There are known to be non-deterministic issues for RNN functions of CUDA. Therefore, when using our saved model weights, there might be some variations in the validation/test accuracy and confusion matrix.

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
     `!python evaluate.py --checkpoint "./alexnet_128.pt" --arch alexnet`  
     where the `arch` parameter has to match the architecture of the saved model.
     
     Optional parameters:  
     `--dataset`: choose dataset to evaluate on -- `validation` or `test`   
     `--batch`: set batch size    
     `--cuda`: enable cuda   

3. `predict.py`: Predicts the class of a video using the trained model.  

    After training the model and saving it, we can use the model to predict the class of handwash videos.

    Run the following command:  
    `%run predict.py --checkpoint "./alexnet_128.pt" --video_path video_file_path`  
    where `video_file_path` is the file path to the video. 
    
    Optional parameters:  
    `--arch`: set architecture (either `convlstm` or `alexnet` or `resnet50` or `custom`)      
    `--cuda`: enable cuda   

    
## Experiments -- Description of notebooks 
The following notebooks are for experiments to find the best hyperparameters and data augmentation.
1. `Model Experiments.ipynb` : Training and experiments are done here.  
    a. Tested out 4 different architectures ConvLSTM and CNN-LSTM with AlexNet, ResNet-50 and custom.  
    b. After finding the best model amongst the 4 architectures, apply hyperparameter tuning by varying the batch size, learning rate and spatial dimensions.  
    c. Once we have derived the best parameters to use with the best model, apply data augmentation such as contrast and translation.  

2. `Model Experiments Testing.ipynb` : Testing of experimental models on validation and test sets. 
- Follow the instructions in the `Model Experiments Testing.ipynb` notebook if you would like to download the experimental models and retest them on the validation set. 

## Demo
Hosted on [Heroku](https://www.heroku.com) and powered by [StreamLit](https://streamlit.io/).   
[Demo](https://handwashdl.herokuapp.com/)
