import argparse
from model import *
from model2 import * 
from dataset import *
from utils import *
import torch
from torch.utils.data import DataLoader

# arguments to command line
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--approach", type=str, default="CNNLSTM", help="choose approach -- ConvLSTM or CNNLSTM")
parser.add_argument("--dataset", type=str, default="validation", help="choose dataset to evaluate on -- validation or test")
parser.add_argument("--batch", type=int, default=4, help="set batch size")
parser.add_argument("--model_dir", type=str, default=None, help="file path to save the model")
parser.add_argument("--arch", type=str, default="custom", help="set architecture")
parser.add_argument("--confusionMatrix", type=bool, default=True, help="print confusion matrix")
parser.add_argument("--cuda", type=bool, default=True, help="enable cuda training")

args = parser.parse_args()
dataset = args.dataset
confusionMatrix=args.confusionMatrix
approach = args.approach
model_dir = args.model_dir
arch  = args.arch
cuda=args.cuda
batch_size = args.batch

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if approach == "CNNLSTM":
    model = build_CNNLSTM_model(arch)
elif approach == "ConvLSTM":
    model = build_ConvLSTM_model(arch)    
else: 
    raise Exception("Invalid approach")
model.to(device)

model = load_model(model, model_dir)

# Dataset
if dataset=="validation":
    ds = Handwash_Dataset('val')
elif dataset=="test":
    ds = Handwash_Dataset('test')
else:
    raise Exception("no such dataset")

# Dataloader
if dataset=="validation":
    loader = DataLoader(ds, batch_size, shuffle=True)
elif dataset=="test":
    loader = DataLoader(ds, 1, shuffle=False)

model.eval()

loss, acc,cm = evaluate(model, device, loader)
print(dataset,'\nLoss: {:.4f} - Accuracy: {:.1f}%\n'.format(loss, acc))
if confusionMatrix:
    print(f"Confusion Matrix:\n{cm}")
    