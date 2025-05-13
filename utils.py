import matplotlib.pyplot as plt
import random
from dataset import CarDataset
import config as cfg
import numpy as np

def random_visual(dataset, class_names, num_images=20, cols=5):
    rows = (num_images + cols - 1) // cols
    indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(16, 10))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]        
        label = int(label)
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

dataset = CarDataset(root=cfg.root, transforms=False, shuffle=True)
class_names = {v: k for k, v in dataset.class_to_idx.items()}

def visualize():
    return random_visual(dataset, class_names)

def plot(trn_losses, val_losses, trn_acc, val_acc):
    epochs = np.arange(1, len(trn_losses)+1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(epochs, trn_losses, label='Train Loss')
    ax[0].plot(epochs, val_losses, label='Val Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Losses')
    ax[0].set_title('Train and Validation Losses')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, trn_acc, label='Train Acc')
    ax[1].plot(epochs, val_acc, label='Val Acc')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracies')
    ax[1].set_title('Train and Validation Accuracies')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()