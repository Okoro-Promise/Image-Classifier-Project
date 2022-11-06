from get_input_args import  get_input_args

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from collections import OrderedDict


# get model inputs
args = get_input_args()
dropout = args.dropout

def model(arch, hidden_units, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Define model arch
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088
    else:
        model = models.resnet18(pretrained=True)
        num_in_features = 512
        
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #Build classifier
    classifer = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(num_in_features, hidden_units)),
        ("relu", nn.ReLU()),
        ("dropout", nn.Dropout(p = dropout)),
        ("fc2", nn.Linear(hidden_units, 102)),
        ("output", nn.LogSoftmax(dim = 1))
    ]))
    # Change model classifer 
    if arch == "vgg16":
        model.classifier = classifer
    else:
        model.fc = classifer
    # connect to device
    model.to(device)
    
    return model

