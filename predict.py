import argparse
from model import *
from utils import *

# arguments to command line
parser = argparse.ArgumentParser(description="Predict action")
parser.add_argument("--arch", type=str, default="alexnet", help="set architecture")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load model")
parser.add_argument("--video_path", type=str, default = None, help="Video path")
parser.add_argument("--cuda", type=bool, default=False, help="enable cuda training")

# get arguments
args = parser.parse_args()
arch = args.arch
checkpoint = args.checkpoint
video_path = args.video_path
cuda=args.cuda

# set cpu / gpu
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = build_model(arch, device)
model.to(device)

model = load_model(model, checkpoint)

output = predict(model, video_path)

print("Prediction:", output)