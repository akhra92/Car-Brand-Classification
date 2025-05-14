import torch
import torch.nn as nn
from model import CustomModel
from torch.optim import Adam
import config as cfg
import numpy as np
from dataset import get_loaders
from transforms import get_transforms


def train():
    model = CustomModel(3, 16).to(cfg.device)
    transforms = get_transforms(train=True)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    train_loader, test_loader = get_loaders(root=cfg.root, transforms=transforms, batch_size=cfg.batch_size)

    trn_losses, val_losses = [], []
    trn_acc, val_acc = [], []

    best_val_acc = 0.0
    best_model_path = 'best_model.pth'

    for epoch in range(cfg.num_epochs):
        print(f'Epoch {epoch+1}/{cfg.num_epochs}')
        trn_batch_loss, val_batch_loss = [], []
        trn_batch_acc, val_batch_acc = [], []
        
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(cfg.device), y.to(cfg.device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            _, predicted = torch.max(y_pred, 1)
            acc = (predicted == y).float().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            trn_batch_loss.append(loss.item())
            trn_batch_acc.append(acc.item())
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(cfg.device), y.to(cfg.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                _, predicted = torch.max(y_pred, 1)
                acc = (predicted == y).float().mean()
                
                val_batch_loss.append(loss.item())
                val_batch_acc.append(acc.item())
        
        epoch_trn_loss = np.mean(trn_batch_loss)
        epoch_trn_acc = np.mean(trn_batch_acc)
        epoch_val_loss = np.mean(val_batch_loss)
        epoch_val_acc = np.mean(val_batch_acc)
        
        scheduler.step(epoch_val_loss)
        
        trn_losses.append(epoch_trn_loss)
        trn_acc.append(epoch_trn_acc)
        val_losses.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        print(f'Train Loss: {epoch_trn_loss:.4f} | Train Acc: {epoch_trn_acc:.4f} |'
            f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}')

        return trn_losses, val_losses, trn_acc, val_acc