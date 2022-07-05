from django.shortcuts import render
from django.http import HttpResponse
import urllib.request
import urllib.parse
import io
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms


# Create your views here.

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(250,250)), 
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            
        ])


def index(request, imagelink):
    import pdb; pdb.set_trace()
    query_string : str = request.META["QUERY_STRING"]
    path : str = (request.path + "?" + query_string).replace("/image/", "")
    if path[0] == "/":
        path = path[1:]
    urllib.request.urlretrieve(path, "image.png")
    picture = Image.open("image.png")
    picture = transform(picture)
    picture = picture.unsqueeze(0)

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("../CNN/model.pth"))
    model.eval()

    return HttpResponse(model.forward(picture))