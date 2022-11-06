import json
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models

from get_input_args import  get_test_inputs
from image_processor import Preprocess



# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    _model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    _model.input_size = checkpoint['input_size']
    _model.output_size = checkpoint['output_size']
    _model.learning_rate = checkpoint['learning_rate']
    _model.classifier = checkpoint['classifier']
    _model.epochs = checkpoint['epochs']
    _model.load_state_dict(checkpoint['state_dict'])
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.optimizer = checkpoint['optimizer']
    
    return _model


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    args = get_test_inputs()
    gpu = args.gpu
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    img_obj = Preprocess()
    img_file = Image.open(image_path)
    
    # use the function 'process_image' to process the image 
    image = img_obj.process_image(img_file)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()
    
    # use model for prediction
    image = image.to(device)
    model.to(device)
    
    #feed forward to model
    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps).data
        class_prob = ps.topk(topk, dim=1)
        top_ps, top_class = ps.topk(topk, dim=1)
        top_ps = np.array(top_ps)[0]
        top_class = np.array(top_class)[0]
        return top_ps, top_class, class_prob



def main ():
    args = get_test_inputs()
    checkpoint = args.checkpoint
    image_path = args.image_path
    top_k = args.top_k
    category_names = args.category_names
    # Load checkpoint
    model = load_checkpoint(checkpoint)
    
    # use model for prediction
    probs, classes, class_prob = predict(image_path, model, topk = top_k)
    
    #Return label name
    with open(category_names, 'r') as json_file:
        name = json.load(json_file)
    labels = [name[str(index + 1)] for index in np.array(class_prob[1][0])]
    print(f"Results for your File: {image_path}")
    print(f"The top {top_k} classes are: {classes}")
    print(f"The top {top_k} probabilities are {probs}")
    print()

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], probs[i]))


if __name__ == "__main__":
    main()
