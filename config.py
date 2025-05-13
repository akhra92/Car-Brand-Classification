import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 30
lr = 1e-3
root = './datasets/car_brands/'
batch_size = 32
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
