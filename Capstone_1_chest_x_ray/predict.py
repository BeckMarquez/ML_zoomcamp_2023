from flask import Flask, request
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path='./chest_x_ray.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

app = Flask('Pneumonia prediction')

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json(force=True)
    img = request_data['img']
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
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    
    return ("The prediction is: {}".format(['normal', 'pneumonia'][pred.argmax()]))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)