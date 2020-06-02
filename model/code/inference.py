import json
import logging
import os
import requests

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from mobilenetv2_cifar10 import MobileNetV2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    """
    Loads the pretrained mobilenetv2 model, works as the entry point for the sagemaker API

    Args:
        model_dir (string): the absolute path of the directory holding the saved .pt
        file of the loading model

    Rets:
        (DataParallel): the pretrained and loaded mobilenetv2 model
    """
    logger.info('Loading model...')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2()

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    logger.info('Current device: {}'.format(device))

    with open(os.path.join(model_dir, 'mobilenet_v2.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()

    logger.info('Loading complete.')

    return model.to(device)


def input_fn(request_body, request_content_type):
    """
    Deserializes JSON encoded data into a torch.Tensor
    
    Args:
        request_body (buffer): a single json list compatible with the loaded model
                               or a 
        requested_content_type (string): specifies type input data

    Rets:
        (Compose): A transformed tensor ready to be passed to predict_fn
    """
    logger.info('Deserializing the input data...')
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Loading image: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)

    elif request_content_type == 'image/*':
        image_data = request_body

    else:
        raise Exception('Unsupported input type')

    # normalize the image data
    image_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return image_transform(image_data) 


def predict_fn(input_data, model):
    """
    Takes the input object and performs an inference agianst the loaded model.

    Args:
        input_data (Compose): a transformed tensor object from input_fn
        model (DataParallel): a pretrained and loaded mobilenetv2 model from model_fn
    
    Rets:
        (Tensor): a torch.Tensor object containing the predition
    """
    logger.info('Performing inference based on the input parameter...')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        output = model(input_data)
        predictions = torch.exp(output)
    
    return predictions


def output_fn(prediction_output, output_content_type):
    """
    
    """
