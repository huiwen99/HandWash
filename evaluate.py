import argparse
from model import *
from dataset import *
from utils import *
import torch
from torch.utils.data import DataLoader

# arguments to command line
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--model_dir", type=str, default=None, help="file path to save the model")
parser.add_argument("--arch", type=str, default="custom", help="set architecture -- convlstm, alexnet, resnet50, custom")
parser.add_argument("--confusionMatrix", type=bool, default=True, help="print confusion matrix if set to True")
parser.add_argument("--cuda", type=bool, default=True, help="enable cuda")

args = parser.parse_args()
model_dir = args.model_dir
arch  = args.arch
confusionMatrix=args.confusionMatrix
cuda=args.cuda

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model(arch) 
model.to(device)

model = load_model(model, model_dir)

# dataset and dataloader
ds = Handwash_Dataset('test')
loader = DataLoader(ds, 1, shuffle=False)

model.eval()

loss, acc,cm = evaluate(model, device, loader)
print('\nLoss: {:.4f} - Accuracy: {:.1f}%\n'.format(loss, acc))
if confusionMatrix:
    print(f"Confusion Matrix:\n{cm}")
    