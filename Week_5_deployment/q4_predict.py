import pickle
from flask import Flask
from flask import request
from flask import jsonify

def load(filename: str):
    with open(filename, 'rb') as f_n:
        return pickle.load(f_n)

dv = load('dv.bin')
model = load('model1.bin')

app = Flask('Credit score')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    score = float(y_pred)

    return jsonify(score)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)