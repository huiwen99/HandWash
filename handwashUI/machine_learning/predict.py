import sys
from model import *
from utils import *

def main(argv):
    compiledOutput = []

    # Change Saved Model Heere
    savedModelPath = './machine_learning/model/alexnet_aug.pt'
    arch = 'alexnet'
    argv = argv[0].split(',')
    
    # Set CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(arch)
    model.to(device)

    model = load_model(model, savedModelPath)

    for eachVideoPath in argv:
        initFileName = eachVideoPath.split('/')[-1].split('.')[0]
        output = predict(model,device, eachVideoPath)
        tempOutput = str(initFileName+':'+output)
        compiledOutput.append(tempOutput)
    
    print(compiledOutput)

if __name__ == "__main__":
    main(sys.argv[1:])
