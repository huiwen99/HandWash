import argparse
from model import *
from utils import *

# arguments to command line
parser = argparse.ArgumentParser(description="Predict action")
parser.add_argument("--arch", type=str, default="alexnet", help="set architecture")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load model")
parser.add_argument("--video_path", type=str, default = None, help="Video path")

# get arguments
args = parser.parse_args()
arch = args.arch
checkpoint = args.checkpoint
video_path = args.video_path

# set cpu
device = torch.device("cpu")

model = build_model(arch)
model.to(device)

model = load_model(model, checkpoint)

output = predict(model,device, video_path)

print(output)