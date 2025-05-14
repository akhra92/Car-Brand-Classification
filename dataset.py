import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
random.seed(2025)
torch.manual_seed(2025)

class CarDataset(Dataset):
    def __init__(self, root, transforms=None, shuffle=False):
        self.root = root
        self.transforms = transforms
        self.shuffle = shuffle
        
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        
        for idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(root, class_name)
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(idx)
        
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)
        
        self.num_classes = len(self.class_to_idx)
        self.task_type = f'Class numbers are {self.num_classes}: Binary Classification' if self.num_classes == 2 else f'Class numbers are {self.num_classes}: Multiclass Classification'
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, torch.tensor(label).float().long()
    

def get_loaders(root, batch_size, transforms):
    dataset = CarDataset(root=root, transforms=transforms, shuffle=True)

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader