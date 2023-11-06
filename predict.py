import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open('model_RF.bin', 'rb') as f_n:
    dv, model = pickle.load(f_n)

app = Flask('Airline customer satisfaction prediction')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict(X)

    result = {
        '1 - satisfied, 0 - unsatisfied. Probability of customer satisfaction': float(y_pred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)