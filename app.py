#-*- coding:utf-8 -*-
import os
from flask import Flask, render_template, jsonify, request, url_for
from werkzeug.utils import secure_filename
import torch
import numpy as np
import torchvision

from torchvision import transforms

from torch import nn, optim

from PIL import Image
import os
from torchvision.models import resnet50

import dbModule

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/images'
device = torch.device('cpu')
model = resnet50(pretrained=False).to(device)
model.fc = nn.Linear(41, 3)
model = model.to(device)



########## 모델 불러오기

device = torch.device('cpu')
model = torch.load(os.getcwd()+'/pt/model_re112.pt', map_location=device)
app = Flask(__name__, static_folder='static', template_folder='templates')


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


@app.route('/contact')  ## 수정 해야함
def contact():
    return render_template('contact.html')


@app.route('/croplist')  ## 수정 해야함
def croplist():
    return render_template('croplist.html')


# @app.route('/our_service', methods = ['POST'])## 수정 해야함 사용자 정보용?
# def get_information():

@app.route('/test', methods=['POST'])  ## 수정 해야함
def fileupload():
    # print(os.getcwd())
    plant = request.form['plant']
    file = request.files['myfile']
    filename = file.filename
    print(filename, plant)
    file.save(os.path.join(os.getcwd()+"/static/images/", filename))
    # file.save('static/images', secure_filename(file.filename))
    #img_src = url_for('static', filename='static/images/' + filename)
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image = Image.open(os.getcwd()+"/static/images/" + filename)
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

    # print(image)
    #
    # trans = transforms.Compose([transforms.Resize(new_shape), transforms.ToTensor(), ])
    # image = torchvision.datasets.ImageFolder(
    #     root='C:/Users/seeum/PycharmProjects/flask/static/images',
    #     transform=trans)



    labels = int(pred)
    if not labels:
        labels = 42
    bug_dict = ['점박이응애', '담배거세미나방', '파밤나방', '목화진딧물', '아메리카잎굴파리', '복숭아혹진딧물', '꽃노랑총채벌레', '대만총채벌레', '명주달팽이', '온실가루이', '차응애', '뽕나무깍지벌레', '풀색노린재', '도둑나방', '알락수염노린재','싸리수염진딧물','썩덩나무노린재','조팝나무진딧물','거세미나방','갈색날개매미충','차먼지응애','벼룩잎벌레','파총채벌레','애모무늬잎말이나방','감자수염진딧물','오이총채벌레','작은뿌리파리','조명나방','미국흰불나방','톱다리개미허리노린재','미국선녀벌레','담배나방','멸강나방','갈색날개노린재','양배추가루진딧물','목화바둑명나방','담배가루이','가루깍지벌레','꽈리허리노린재', "없는 결과"]
    bug = bug_dict[labels-1]
    db_class = dbModule.Database()
    sql = 'SELECT * FROM list WHERE plant = "{}" AND name = "{}" limit 5;'.format(plant, bug)
    print(sql)
    row = db_class.executeALL(sql)
    print(row)
    return render_template('result.html', bug=bug, data=row, filename=filename, plant=plant)

#
#
# @app.route('/result', methods=['POST'])  ## 수정 해야함
# def result():
#     print



# def get_image():
#     f = request.files['myfile']
#     f.save('C:/Users/seeum/PycharmProjects/flask/static/images', 'new')
#
#     new_shape = (128, 128)
#     image = cv2.imread('C:/Users/seeum/PycharmProjects/flask/static/images/new.jpg', cv2.IMREAD_COLOR)
#     trans = transforms.Compose([transforms.Resize(new_shape), transforms.ToTensor(), ])
#     image = np.array(image)
#     image = trans(image=image)['image']
#     image = image.unsqueeze(0)
#     # image = torchvision.datasets.ImageFolder(root='C:/Users/seeum/PycharmProjects/flask/static/images',
#     #                                          transform=trans)
#     return predicted_value(image)

# @app.route('')



# # #for model
# model = resnet50()
# model.load_state_dict(torch.load('mnist_model.pt'), strict=False)
# model.eval()

# normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# @app.route('/inference', methods=['POST'])
# def inference():
#     data = request.json
#     _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
#     return str(result.item())


app.run(host='0.0.0.0', port=5001)
# app.config['UPLOAD_FOLDER'] = '/static/images'