from django.shortcuts import render
from django.http import HttpResponse
import urllib.request
import urllib.parse
import io
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn


# Create your views here.

def index(response, imagelink):
    imagelink = urllib.parse.unquote(imagelink)
    path = io.StringIO(urllib.request.urlopen(imagelink).read())
    picture = Image.open(path)

    # 1. Design the model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("../CNN/model.pth"))
    model.eval()

    return HttpResponse(model.forward(picture))