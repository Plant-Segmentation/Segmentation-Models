import os
import json
import argparse
import dataloaders
import models
import numpy as np

import torch

from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss_weight = config['class_weight'] if 'class_weight' in config else None
    loss = getattr(losses, config['loss'])(weight=loss_weight,
                                           ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config['trainer']['epochs'] = 150

    main(config, args.resume)
