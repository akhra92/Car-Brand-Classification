import torchvision.transforms as T

def get_transforms(train=False):
    trn_tfs = T.Compose([T.Resize((224, 224)),
                         T.RandomHorizontalFlip(p=0.3),
                         T.RandomRotation(degrees=15),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    val_tfs = T.Compose([T.Resize((224,224)),                        
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
    
    if train:
        return trn_tfs
    else:
        return val_tfs