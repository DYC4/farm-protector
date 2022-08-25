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

app = Flask(__name__)
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
convolutional_network = resnet50(pretrained= True) ###  수정   ###
convolutional_network.fc = nn.Flatten() #### fc layer 사용
model = PrototypicalNetworks(convolutional_network).to(device)
model = torch.load('C:\\Users\\tjtnd\\PycharmProjects\\helloFlask\\pt\\model_re112.pt', map_location=device)
