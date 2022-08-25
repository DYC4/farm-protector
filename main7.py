from flask import Flask, render_template, jsonify, request, url_for
from werkzeug.utils import secure_filename
import torch
import numpy as np
import torchvision

from torchvision import transforms
from torchvision.models import resnet50

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

from pathlib import Path
from torch.utils.data import Dataset

# from torchvision.io import read_image
import cv2

from PIL import Image
import os
from torchvision.models import resnet50

app = Flask(__name__, static_folder='static', template_folder='templates')

device = torch.device('cpu')
model = resnet50(pretrained=False).to(device)
model.fc = nn.Linear(41, 3)
model = model.to(device)


model = torch.load('C:/Users/seeum/Downloads/model_transfer.pt')




@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/we_do')  ## 수정 해야함
def we_do():
    return render_template('intro1.html')


@app.route('/we_are')  ## 수정 해야함
def we_are():
    return render_template('intro2.html')


@app.route('/our_service')  ## 수정 해야함
def our_service():
    return render_template('upload.html')


# @app.route('/our_service', methods = ['POST'])## 수정 해야함 사용자 정보용?
# def get_information():

@app.route('/fileupload', methods=['POST'])  ## 수정 해야함
def fileupload():
    file = request.files['myfile']
    filename = file.filename
    file.save(os.path.join('static/images/', filename))
    img_src = url_for('static', filename='static/images/' + filename)
    image = Image.open('C:/Users/seeum/PycharmProjects/flask/static/images/'+filename)
    trans = transforms.Compose([transforms.RandomResizedCrop(84),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.480, 0.532, 0.340], [0.166, 0.159, 0.160])
                                ])

    image = trans(image)
    image = image.unsqueeze(0)
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)
    pred = pred +1

    return render_template('result.html', bug = int(pred))

#
#
app.run(host='0.0.0.0', port=81)





