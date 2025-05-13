import random
import seaborn as sns
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import config as cfg


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add(m)
        return tensor
    
class ModelInferenceVisualizer:
    def __init__(self, model, device, class_names=None, im_size=224, mean=mean, std=std):
        self.denormalize = Denormalize(mean, std)
        self.model = model
        self.device = device
        self.class_names = class_names
        self.im_size = im_size
        self.model.eval()
        
    def tensor_to_image(self, tensor):        
        tensor = tensor.detach().cpu()
        tensor = self.denormalize(tensor)
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = np.clip(tensor, 0, 1)
        return (tensor * 255).astype(np.uint8)
    
    def generate_cam_visualization(self, image_tensor):
        cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.conv4], use_cuda=self.device=='cuda')
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0, :]
        return grayscale_cam
    
    def infer_and_visualize(self, test_loader, num_images=20, cols=5):
        preds, images, labels = [], [], []
        accuracy, count = 0, 1
        rows = (num_images + cols - 1) // cols
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader), desc='Inference'):
                x, y = batch
                x, y = x.to(cfg.device), y.to(cfg.device)
                y_pred = self.model(x)
                _, predicted = torch.max(y_pred, 1)
                accuracy += (predicted == y).sum().item()
                images.append(x[0])
                preds.append(predicted[0].item())
                labels.append(y[0].item())
                
        print(f'Accuracy of the model on the test data -> {(accuracy / len(test_loader.dataset)):.4f}')
        
        plt.figure(figsize=(20, 10))
        indices = random.sample(range(len(images)), num_images)
        for idx, index in enumerate(indices):
            img = self.tensor_to_image(images[index])
            pred_idx = preds[index]
            label_idx = labels[index]
            
            plt.subplot(rows, cols, count)
            count += 1
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            grayscale_cam = self.generate_cam_visualization(images[index])
            visualization = show_cam_on_image(img / 255.0, grayscale_cam, image_weight=0.4, use_rgb=True)
            plt.imshow(cv2.resize(visualization, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet')
            plt.axis('off')
            
            if self.class_names:
                gt_name = self.class_names[label_idx]
                pred_name = self.class_names[pred_idx]
                color = 'green' if gt_name == pred_name else 'red'
                plt.title(f'GT -> {gt_name}; PRED -> {pred_name}', color=color)
                
        plt.figure(figsize=(20, 10))
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm,annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.show()