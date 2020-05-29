import torch
import torch.nn as nn
import logging
import os
from mobilenetv2_cifar10 import MobileNetV2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):

    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2()

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    logger.info('Current device: {}'.format(device))

    with open(os.path.join(model_dir, 'mobilenet_v2.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)
