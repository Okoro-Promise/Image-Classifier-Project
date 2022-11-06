import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

from workspace_utils import active_session
from get_input_args import  get_input_args
from image_processor import Preprocess
from model import model


args = get_input_args()
lr = args.learning_rate
n_epochs = args.epochs
arch = args.arch
hidden_units = args.hidden_units
gpu = args.gpu
data_dir = args.data_dir

preprocess = Preprocess()
traindataloader = preprocess.process_train(data_dir + "/train")
validdataloader = preprocess.process_valid(data_dir + "/valid")

model =  model(arch, hidden_units, gpu)

# Set criterion and optimzer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters() if arch == "vgg16" else model.fc.parameters(), lr = lr)
# Train the network
def main():
    
    with active_session():
        epochs = n_epochs
        steps = 0
        running_loss = 0
        print_every = 10
        #change to cuda
        if gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        for epoch in range(epochs):
            for images, labels in traindataloader:
                steps += 1

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    val_loss = 0
                    accuracy = 0

                    for images, labels in validdataloader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        with torch.no_grad():
                            logps = model(images)
                            loss = criterion(logps, labels)
                            val_loss += loss.item()

                            # Calculate the accuracy
                            ps = torch.exp(logps).data
                            top_ps, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor))

                    print(f"Epoch {epoch + 1}/{epochs}...."
                         f"Train Loss: {running_loss /print_every:.3f}..."
                         f"Test Loss: {val_loss / len(validdataloader):.3f}...."
                         f"Test Accuracy: {accuracy / len(validdataloader):.3f}")
                    running_loss = 0
                    model.train()
                    
     # Create checkpoint  
    train_transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.Resize(224),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])

                                    ])
    train_dataset = datasets.ImageFolder(data_dir + "/train", transform = train_transforms)
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'input_size': (25088 if arch == "vgg16" else 512) ,
            'output_size': 102,
            'arch': ('vgg16' if arch == "vgg16" else "resnet18"),
            'learning_rate': lr,
            'classifier': (model.classifier if arch == "vgg16" else model.fc),
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    print("Checkpoint Saved...")
    
    
if __name__ == "__main__":
    main()
