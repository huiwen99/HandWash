import argparse
from model import *
from dataset import *
from utils import *
import torch
from torch.utils.data import DataLoader

# arguments to command line
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--dataset", type=str, default="test", help="choose dataset to evaluate on -- validation or test")
parser.add_argument("--batch", type=int, default=32, help="set batch size")
parser.add_argument("--checkpoint", type=str, default=None, help="file path to save the model")
parser.add_argument("--arch", type=str, default="alexnet", help="set architecture -- convlstm, alexnet, resnet50, custom")
parser.add_argument("--cuda", type=bool, default=False, help="enable cuda training")

args = parser.parse_args()
dataset = args.dataset
checkpoint = args.checkpoint
arch  = args.arch
cuda=args.cuda
batch_size = args.batch

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model(arch, device) 
model.to(device)

model = load_model(model, checkpoint)

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
print(f"Confusion Matrix:\n{cm}")