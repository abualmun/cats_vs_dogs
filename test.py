import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
import sys
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

model = models.resnet18()
model.fc = nn.Linear(in_features=model.fc.in_features,out_features=2)
model.load_state_dict(torch.load('./cats_vs_dogs_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

from PIL import *
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = image.unsqueeze(0) 
    return image

image_path = sys.argv[1]
image_view = mpimg.imread(image_path)
image = image_loader(image_path) 

output = model.forward(image)
_, predicted = torch.max(output, 1)
plt.imshow(image_view)

if predicted.item() == 0:
    print("It's a Cat!")
    plt.title("It's a Cat!")
else:
    print("It's a Dog!")
    plt.title("It's a Dog!")
plt.show()