from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model
def load_model():
    with open('model/gradient_boosting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([[data['crudeOilPrice'], data['production'], data['crudeOilExport']]])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
