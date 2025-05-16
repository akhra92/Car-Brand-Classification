import torch
import onnx
from model import CustomModel
import config as cfg
import numpy as np
from PIL import Image
from torchvision import transforms as T
import onnxruntime as ort


def run_onnx():

    preprocess = T.Compose([T.Resize((224,224)),                        
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    model = CustomModel(in_channels=3, num_classes=16)
    model.load_state_dict(torch.load('./saved_models/best_model.pth'))
    model.eval

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        'model.onnx',
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    onnx_model = onnx.load('model.onnx')
    onnx.checker.check_model(onnx_model)
    print('ONNX model is ready!')

    onnx_model_path = 'model.onnx'
    session = ort.InferenceSession(onnx_model_path)

    image_path = './sample_images/image25.jpg'
    classes = ['BMW', 'Byd', 'Chevrolet', 'Ford', 'Honda', 'Hundai', 'Mercedes-Benz', 'Mitsubishi', 'Renault', 'Skoda', 'Suzuki', 'Toyota', 'kia', 'lada', 'nissan', 'volkswagen']
    image = Image.open(image_path).convert('RGB')

    input_tensor = preprocess(image).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    input_name = session.get_inputs()[0].name
    print(input_name)

    output = session.run(None, {input_name: input_numpy})

    output_array = output[0]
    predicted_class = np.argmax(output_array, axis=1)
    predicted_class_name = classes[predicted_class[0]]
    print(f'Predicted class name: {predicted_class_name}')


if __name__ == '__main__':
    run_onnx()