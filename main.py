import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import random
import matplotlib.pyplot as plt
import config as cfg
from dataset import get_loaders
from model import CustomModel
from train import train
from utils import visualize, plot
from test import ModelInferenceVisualizer
from transforms import get_transforms

random.seed(2025)
torch.manual_seed(2025)

transforms = get_transforms(train=False)
train_loader, test_loader = get_loaders(root=cfg.root, transforms=transforms, batch_size=cfg.batch_size)

def main():
    visualize()
    trn_losses, val_losses, trn_acc, val_acc = train()
    plot(trn_losses, val_losses, trn_acc, val_acc)
    classes = ['BMW', 'Byd', 'Chevrolet', 'Ford', 'Honda', 'Hundai', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Skoda', 'Suzuki', 'Toyota', 'kia', 'lada', 'nissan', 'volkswagen']
    model = CustomModel(3, 16).to(cfg.device)
    model.load_state_dict(torch.load('./saved_models/best_model.pth'))
    inference_visualizer = ModelInferenceVisualizer(model=model,
                                                    device=cfg.device,
                                                    class_names=classes,
                                                    im_size=cfg.im_size)
    inference_visualizer.infer_and_visualize(test_loader,num_images=20, cols=5)

if __name__ == '__main__':
    main()

