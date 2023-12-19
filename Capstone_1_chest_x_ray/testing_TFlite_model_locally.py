import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='./chest_x_ray.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

source = ['./img_for_test/person6_bacteria_22.jpeg', 
       './img_for_test/person19_virus_50.jpeg',
       './img_for_test/NORMAL2-IM-0030-0001.jpeg',
       './img_for_test/NORMAL2-IM-0095-0001.jpeg']

classes = ['normal', 'pneumonia']

def img_prep(source):
    with Image.open(source) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.NEAREST)
    X = np.array(img, dtype='float32')
    X = np.array([X])
    X /= 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    X[..., 0] -= mean[0]
    X[..., 1] -= mean[1]
    X[..., 2] -= mean[2]
    X[..., 0] /= std[0]
    X[..., 1] /= std[1]
    X[..., 2] /= std[2]
    return X

def predict(source):
    X = img_prep(source=source)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return dict(zip(classes, pred[0]))

for elem in source:
    print(predict(elem))