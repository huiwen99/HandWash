import argparse
from model import *
from dataset import *
from utils import *
import torch
from torch.utils.data import DataLoader

# arguments to command line
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--checkpoint", type=str, default=None, help="file path to save the model")
parser.add_argument("--arch", type=str, default="custom", help="set architecture -- convlstm, alexnet, resnet50, custom")
parser.add_argument("--confusion_matrix", type=bool, default=True, help="print confusion matrix if set to True")

args = parser.parse_args()
checkpoint = args.checkpoint
arch  = args.arch
confusion_matrix=args.confusion_matrix

# set cpu
device = torch.device("cpu")

model = build_model(arch) 
model.to(device)

model = load_model(model, checkpoint)

# dataset and dataloader
ds = Handwash_Dataset('test')
loader = DataLoader(ds, 1, shuffle=False)

model.eval()

loss, acc, cm = evaluate(model, device, loader)
print('\nLoss: {:.4f} - Accuracy: {:.1f}%\n'.format(loss, acc))
if confusion_matrix:
    print(f"Confusion Matrix:\n{cm}")
    