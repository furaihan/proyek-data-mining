from flask import Flask, request, jsonify
from preprocessing import process_input
import joblib
import nltk
nltk.download('punkt')

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
        return jsonify({
            'error': 'Input data must contain a key named "headline"',
            'status': 'error'
        }), 400
    
    headline = data['headline']
    processed_data = process_input(headline)
    
    # Get probability estimates
    probabilities = model.predict_proba([processed_data])[0]
    
    # Assuming binary classification where index 1 corresponds to clickbait
    clickbait_probability = float(probabilities[1])
    
    is_clickbait = clickbait_probability > 0.5
    
    result = {
        'prediction': 'clickbait' if is_clickbait else 'not_clickbait',
        'is_clickbait': bool(is_clickbait),
        'headline': headline,
        'clickbait_probability': clickbait_probability,
        'status': 'success'
    }
    
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
    