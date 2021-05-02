import sys
from model import *
from utils import *

def main(argv):
    compiledOutput = []

    # Change Saved Model Heere
    savedModelPath = './machine_learning/model/alexnet_128.pt'
    arch = 'alexnet'
    argv = argv[0].split(',')
    # Set CPU
    device = torch.device('cpu')

    model = build_model(arch)
    model.to(device)

    model = load_model(model, savedModelPath)

    for eachVideoPath in argv:
        initFileName = eachVideoPath.split('/')[-1].split('.')[0]
        output = predict(model, eachVideoPath)
        tempOutput = str(initFileName+':'+output)
        compiledOutput.append(tempOutput)
    
    print(compiledOutput)

if __name__ == "__main__":
    main(sys.argv[1:])
