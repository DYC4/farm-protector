#-*- coding:utf-8 -*-
import easyfsl
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
from torchvision.models import resnet50

from pathlib import Path
from torch.utils.data import Dataset

# from torchvision.io import read_image
# import cv2

from PIL import Image
import os
from torchvision.models import resnet50

import dbModule

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/images'



class CustomDataset(Dataset):
    @staticmethod
    def _validate_root_dir(root):
        # todo: raise exception or warning
        pass

    @staticmethod
    def _validate_train_flag(train: bool, val: bool, test: bool):
        assert [train, val, test].count(True) == 1, "one of train, valid & test must be true."

    def __init__(self, root, train: bool = False, val: bool = False, test: bool = False,
                 transform=None, target_transform=None, ):

        self._validate_root_dir(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.data_dir = Path(root) / 'train'
        elif val:
            self.data_dir = Path(root) / 'val'
        elif test:
            self.data_dir = Path(root) / 'test'

        self._image_paths = sorted(
            list(self.data_dir.glob("**/*.jpg")) +
            list(self.data_dir.glob("**/*.jpeg")) +
            list(self.data_dir.glob("**/*.png")))
        self._image_labels = [int(i.parent.name) for i in self._image_paths]
        assert len(self._image_paths) == len(self._image_labels)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        x = Image.open(str(self._image_paths[idx]))  ## grayscale 을 위한 .convert("L") 삭제
        y = self._image_labels[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(x)
        return x, y

    def get_labels(self):
        return self._image_labels


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


########## 모델 불러오기

device = torch.device('cpu')
#convolutional_network = resnet50(pretrained=True) ###  수정   ###
#convolutional_network.fc = nn.Flatten() #### fc layer 사용
#model = PrototypicalNetworks(convolutional_network).to(device)
model = torch.load('C:\\Users\\tjtnd\\PycharmProjects\\helloFlask\\pt\\model_re112.pt', map_location=device)




N_WAY =40  # Number of classes in a task
N_SHOT = 1  # Number of images per class in the support set
N_QUERY = 1  # Number of images per class in the query set
N_EVALUATION_TASKS = 100
N_VALIDATION_TASKS = 10

image_size = (128, 128)

ds_train = CustomDataset('C:\\Users\\tjtnd\\PycharmProjects\\helloFlask\\pest', train=True,  transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), ]))

train_sampler = TaskSampler(
    ds_train, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

train_loader = DataLoader(
    ds_train,
    batch_sampler=train_sampler,
    # num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,)


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
    file.save(os.path.join("C:\\Users\\tjtnd\\PycharmProjects\\helloFlask\\static\\images\\", filename))
    # file.save('static/images', secure_filename(file.filename))
    #img_src = url_for('static', filename='static/images/' + filename)
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image = Image.open("C:\\Users\\tjtnd\\PycharmProjects\\helloFlask\\static\\images\\" + filename)
    new_shape = (128, 128)
    # image = np.array(image)
    trans = transforms.Compose([transforms.Resize(new_shape), transforms.ToTensor(), ])
    image = trans(image)
    image = image.unsqueeze(0)

    # print(image)
    #
    # trans = transforms.Compose([transforms.Resize(new_shape), transforms.ToTensor(), ])
    # image = torchvision.datasets.ImageFolder(
    #     root='C:/Users/seeum/PycharmProjects/flask/static/images',
    #     transform=trans)

    (
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,

    ) = next(iter(train_loader))

    example_scores = model(
        example_support_images.to(device),
        example_support_labels.to(device),
        image.to(device),
    ).detach()

    _, example_predicted_labels = torch.max(example_scores.data, 1)
    labels = example_class_ids[example_predicted_labels]
    bug_dict = ['점박이응애', '담배거세미나방', '파밤나방', '목화진딧물', '아메리카잎굴파리', '복숭아혹진딧물', '꽃노랑총채벌레', '대만총채벌레', '명주달팽이', '온실가루이', '차응애', '뽕나무깍지벌레', '풀색노린재', '도둑나방', '알락수염노린재','싸리수염진딧물','썩덩나무노린재','조팝나무진딧물','거세미나방','갈색날개매미충','차먼지응애','벼룩잎벌레','파총채벌레','애모무늬잎말이나방','감자수염진딧물','오이총채벌레','작은뿌리파리','조명나방','미국흰불나방','톱다리개미허리노린재','미국선녀벌레','담배나방','멸강나방','갈색날개노린재','양배추가루진딧물','목화바둑명나방','담배가루이','가루깍지벌레','꽈리허리노린재']
    bug = bug_dict[labels]
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


app.run(host='localhost', port=5000)
# app.config['UPLOAD_FOLDER'] = '/static/images'