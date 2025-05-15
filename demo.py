import torch, argparse
import streamlit as st
from transforms import get_transforms  
from PIL import Image, ImageFont
from model import CustomModel
st.set_page_config(layout='wide')


def run(args):        
    
    cls_names = ['BMW', 'Byd', 'Chevrolet', 'Ford', 'Honda', 'Hundai', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Skoda', 'Suzuki', 'Toyota', 'kia', 'lada', 'nissan', 'volkswagen']    
    
    num_classes = len(cls_names)    
    
    tfs = get_transforms(train = False)    
    
    default_path = "./sample_images/chevrolet.jpg"    

    m = load_model(num_classes, args.checkpoint_path)
    st.title("Car Brand Classification")
    file = st.file_uploader('Please upload your image')

    im, out = predict(m = m, path = file, tfs = tfs, cls_names = cls_names) if file else predict(m = m, path = default_path, tfs = tfs, cls_names = cls_names)
    st.write(f"Input Image: ")
    st.image(im)
    st.write(f"Predicted as {out}")


def load_model(num_classes, checkpoint_path): 
    
    m = CustomModel(3, num_classes=num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location = "cpu"))
    
    return m.eval()

def predict(m, path, tfs, cls_names):
    
    fontpath = "SpoqaHanSansNeo-Light.ttf"
    font = ImageFont.truetype(fontpath, 200)
    im = Image.open(path)
    im.save(path)
    
    return im, cls_names[int(torch.max(m(tfs(im).unsqueeze(0)).data, 1)[1])]
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Car Brands Classification Demo')
    
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = 'saved_models/best_model.pth', help = "Path to the checkpoint")
    
    args = parser.parse_args()     
    
    run(args) 