from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import re
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
import logging
from logging.handlers import RotatingFileHandler

# Set NLTK data path
os.environ['NLTK_DATA'] = '/opt/clickbait-api/nltk_data'

# Configure logging
if not os.path.exists('/var/log/clickbait-api'):
    os.makedirs('/var/log/clickbait-api')

logging.basicConfig(
    handlers=[RotatingFileHandler('/var/log/clickbait-api/app.log', maxBytes=100000, backupCount=10)],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)


# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize Sastrawi components
factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stopword_indonesia = stopword_factory.get_stop_words()
stemmer = factory.create_stemmer()

# Constants
angka = ['nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan']
potential_clickbait_words = ['bikin', 'viral', 'gara', 'fakta', 'kejut', 'kamu', 'wajib', 'pakai', 'ala', 'heboh', 'geger', 'video', 'foto', 'cantik']

class StructuredFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert features column to numpy array
        return np.array(X.tolist(), dtype=np.float32)

class TextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Join stemmed text back into strings
        return X.apply(lambda x: ' '.join(x))

# Global variable to store the loaded model
best_pipeline = None
best_params = None

def load_model():
    """Load the trained model on application startup"""
    global best_pipeline, best_params
    
    try:
        print("Loading trained model...")
        # Load the best pipeline
        best_pipeline = joblib.load('best_lgbm_pipeline.joblib')
        
        # Load best parameters (optional)
        try:
            best_params = joblib.load('best_parameters.joblib')
            print("Model and parameters loaded successfully!")
        except:
            print("Model loaded successfully! (Parameters file not found)")
            best_params = None
        
        # Display model info
        print(f"Model type: {type(best_pipeline.named_steps['model']).__name__}")
        print(f"Pipeline steps: {list(best_pipeline.named_steps.keys())}")
        
        return True
        
    except FileNotFoundError:
        print("Error: Model file 'best_lgbm_pipeline.joblib' not found!")
        print("Please ensure you've run the training script and saved the model.")
        return False

def preprocess_text(text):
    """Preprocess text for model input"""
    # remove non-alphabet characters
    text = re.sub(r'[^A-Za-z0-9!?]', ' ', text)
    text = re.sub(r'([!?])', r' \1 ', text)
    # remove whitespace
    text = text.strip()
    # remove newline
    text = text.replace('\n\n', ' ')
    # remove extra space
    text = re.sub(' +', ' ', text)
    # lowercase
    text = text.lower()
    # tokenize
    text = word_tokenize(text)
    # replace punctuation tokens
    text = ['EXCLAMATIONTOKEN' if token == '!' else 'QUESTIONTOKEN' if token == '?' else token for token in text]
    result = text
    return result

def count_full_word_capitals(text):
    """Count total characters in fully capitalized words"""
    return sum(len(word) for word in text.split() if word.isupper() and len(word) > 1)

def count_all_caps_words(text):
    """Count number of fully capitalized words"""
    return sum(1 for word in text.split() if word.isupper() and len(word) > 1)

def check_features(text):
    """Extract features from text"""
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    stemmed_words = [stemmer.stem(word.lower()) for word in words]
    result = []
    
    # Feature 1: contains exclamation mark
    result.append(1 if '!' in text else 0)
    
    # Feature 2: contains question mark
    result.append(1 if '?' in text else 0)
    result.append(1 if '??' in text else 0)
    
    # Feature 3: contains multiple exclamation marks
    result.append(1 if '!!' in text else 0)
    
    # Feature 4: contains multiple dots (ellipsis indicator)
    result.append(1 if '..' in text else 0)
    
    # Feature 5: contains potential clickbait words (after stemming)
    result.append(1 if any(stem in potential_clickbait_words for stem in stemmed_words) else 0)
    result.append(sum(1 for stem in stemmed_words if stem in potential_clickbait_words))
    
    # Feature 6: count of exclamation marks
    result.append(text.count('!'))
    
    # Feature 7: total capital letters from full caps words
    result.append(count_full_word_capitals(text))
    
    # Feature 8: total length of text (in characters)
    result.append(len(text))
    
    # Feature 9: word count
    result.append(word_count)
    
    # Feature 10: average word length
    result.append(avg_word_length)
    
    # Feature 11: starts with number (digit or number word in Indonesian)
    result.append(1 if words and (words[0].isdigit() or words[0].lower() in angka) else 0)
    
    # Feature 12: count of full-uppercase words
    result.append(count_all_caps_words(text))
    
    # Feature 13: ratio of uppercase characters to total characters
    result.append(sum(1 for c in text if c.isupper()) / len(text) if text else 0)
    
    result.append(len(re.findall(r'\d+', text)))
    result.append(int(bool(re.search(r'^\d+', text))))
    result.append(sum(1 for w in words if len(w) <= 2))
    result.append(sum(1 for w in words if len(w) >= 10))
    
    return result

def predict_clickbait(text_input):
    """Make clickbait prediction for given text"""
    if best_pipeline is None:
        return {"error": "Model not loaded", "status": "error"}
    
    try:
        # Create dataframe
        df = pd.DataFrame([text_input], columns=['title'])
        
        # Preprocess text
        df['preprocessed_text'] = df['title'].apply(preprocess_text)
        df['stemmed_text'] = df['preprocessed_text'].apply(lambda x: [stemmer.stem(word.lower()) for word in x])
        df['features'] = df['title'].apply(check_features)
        
        # Make predictions
        prediction = best_pipeline.predict(df[['stemmed_text', 'features']])
        prediction_proba = best_pipeline.predict_proba(df[['stemmed_text', 'features']])
        
        # Handle probability array
        if hasattr(prediction_proba, 'ndim') and prediction_proba.ndim > 1:
            prediction_proba_row = prediction_proba[0]
        else:
            prediction_proba_row = np.array(prediction_proba)
        
        # Get model classes
        classes = best_pipeline.named_steps['model'].classes_
        
        # Determine result
        is_clickbait = int(prediction[0])
        label = "clickbait" if is_clickbait == 1 else "non-clickbait"
        
        # Create result dictionary
        result = {
            'prediction': label,
            'is_clickbait': is_clickbait,
            'clickbait_probability': {str(cls): float(prob) for cls, prob in zip(classes, prediction_proba_row)},
            'headline': text_input,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if best_pipeline is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided', 'status': 'error'}), 400
        
        # Extract text input
        text_input = data.get('text', '').strip()
        
        if not text_input:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400
        
        # Make prediction
        result = predict_clickbait(text_input)
        
        if result.get('status') == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Request processing error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided', 'status': 'error'}), 400
        
        # Extract text inputs
        text_inputs = data.get('texts', [])
        
        if not text_inputs or not isinstance(text_inputs, list):
            return jsonify({'error': 'No valid text list provided', 'status': 'error'}), 400
        
        if len(text_inputs) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 100)', 'status': 'error'}), 400
        
        # Make predictions for each text
        results = []
        for i, text_input in enumerate(text_inputs):
            if isinstance(text_input, str) and text_input.strip():
                result = predict_clickbait(text_input.strip())
                result['batch_index'] = i
                results.append(result)
            else:
                results.append({
                    'batch_index': i,
                    'error': 'Invalid text input',
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Batch processing error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if best_pipeline is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    try:
        model_type = type(best_pipeline.named_steps['model']).__name__
        pipeline_steps = list(best_pipeline.named_steps.keys())
        classes = best_pipeline.named_steps['model'].classes_.tolist()
        
        return jsonify({
            'model_type': model_type,
            'pipeline_steps': pipeline_steps,
            'classes': classes,
            'has_parameters': best_params is not None,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Model info error: {str(e)}',
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed', 'status': 'error'}), 405

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model()
    
    if not model_loaded:
        print("Warning: Model could not be loaded. Service will start but predictions will fail.")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=False,     # Set to True for development
        threaded=True    # Handle multiple requests
    )