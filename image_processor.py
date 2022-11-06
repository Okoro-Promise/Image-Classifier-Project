import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms, models



class Preprocess():
       
    def process_image(self, image):
        
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # TODO: Process a PIL image for use in a PyTorch model
        # Resize the image
        new_size = 256
        width = image.size[0]
        height = image.size[1]

        if height > width:
            height = int(max(height * new_size / width, 1))
            width = int(new_size)
        else:
            width = int(max(width * new_size / height, 1))
            height = int(new_size)

        # Resize image 
        im = image.resize((width, height))

        # Crop out the center 224 x 224
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = left + 224
        bottom = top + 224
        im = im.crop((left, top, right, bottom))

        # convert color channels to 0-1
        # convert from integers to floats
        im = np.array(im)
        im = im.astype('float32')

        # normalize to the range 0-1
        im /= 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std

        im = np.transpose(im, (2, 0, 1))

        return im


    def process_train(self,train_dir):
        
        ''' Scales, crops, and normalizes a PIL image for training a PyTorch model,
        returns an Tensor
        '''
        train_transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.Resize(224),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])

                                    ])
        # Load the datasets with ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
        self.train_dataset = train_dataset
        #Using the image dataset and the trainform, define the dataloader
        traindataloader = torch.utils.data.DataLoader(train_dataset,batch_size= 64, shuffle=True)
        
        return traindataloader
    
    
    def process_valid (self, valid_dir):
          
        ''' Scales, crops, and normalizes a PIL image for training a PyTorch model,
        returns an Tensor
        '''
        
        val_test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

        ])
        # Load the datasets with ImageFolder
        valid_dataset = datasets.ImageFolder(valid_dir, transform = val_test_transforms)
        
        #Using the image dataset and the trainform, define the dataloader
        validdataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle=True)
        
        return validdataloader
        

        