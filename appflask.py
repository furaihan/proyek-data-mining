from flask import Flask, request, jsonify
from preprocessing import process_input
import joblib

MODEL_PATH = 'model.pkl'
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return 'Indonesian Clickbait Headline Detector'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'headline' not in data:
        return jsonify({'error': 'Input data must contain a key named "headline"'}), 400
    headline = data['headline']
    processed_data = process_input(headline)
    prediction = model.predict([processed_data])[0]
    result = 'Headline ini bukan clickbait' if prediction == 0 else 'Headline ini clickbait'
    return jsonify({'prediction': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
    