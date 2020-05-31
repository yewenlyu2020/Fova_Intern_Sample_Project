import torch
import torch.nn as nn
import logging
import os
from mobilenetv2_cifar10 import MobileNetV2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def model_fn(model_dir):
    """
    Load the pretrained mobilenetv2 model, work as the endpoint of the sagemaker API

    Args:
        model_dir (string): the absolute path of the directory holding the saved .pt
        file of the loading model

    Returns:
        (DataParallel): the pretrained mobilenetv2 model
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
