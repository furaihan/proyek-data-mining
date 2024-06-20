import re
import json
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Constants
MODEL_PATH = 'model.pkl'
VOCAB_PATH = 'vocab.json'
PADDING_LENGTH = 27
EXCLAMATION = '!'
QUESTION = '?'
POTENTIAL_CLICKBAIT_WORDS = ['bikin', 'viral', 'gara', 'fakta', 'kejut', 'mengejutkan', 'menghebohkan', 'heboh', 'gempar']
ANGKA = ['nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan']

# Load resources
stemmer = StemmerFactory().create_stemmer()
with open(VOCAB_PATH, 'r') as vocab_file:
    vocab = json.load(vocab_file)

def preprocess_text(text):
    """
    Preprocess the input text by removing non-alphabet characters,
    extra spaces, converting to lowercase, and tokenizing.
    """
    text = re.sub(r'[^A-Za-z0-9?!]', ' ', text).strip().replace('\n', ' ')
    text = re.sub(' +', ' ', text).lower()
    return word_tokenize(text)

def check_features(text):
    """
    Check for specific features in the text such as exclamation marks,
    question marks, digits, numbers, and potential clickbait words.
    """
    text_str = ' '.join(text)
    features = [
        1 if EXCLAMATION in text_str else 2,
        3 if QUESTION in text_str else 4,
        5 if '!!' in text_str else 6,
        7 if any(char.isdigit() for char in text_str) else 8,
        9 if any(word in ANGKA for word in text.lower()) else 10,
        11 if any(word in POTENTIAL_CLICKBAIT_WORDS for word in text.lower()) else 12,
    ]
    return features

def update_vocab(text):
    """
    Update the vocabulary with new words from the text.
    """
    global vocab
    last_index = len(vocab) + 1000
    for word in text:
        if word not in vocab:
            vocab[word] = last_index
            last_index += 1

def vectorize_and_pad(text):
    """
    Vectorize the text using the vocabulary and pad it to the required length.
    """
    vectorized = [vocab[word] for word in text if word in vocab]
    return vectorized[:PADDING_LENGTH] + [-1] * (PADDING_LENGTH - len(vectorized))

def process_input(text):
    """
    Process the user input: preprocess, stem, extract features, update vocabulary,
    vectorize, and pad the text.
    """
    preprocessed = preprocess_text(text)
    stemmed = [stemmer.stem(word) for word in preprocessed]
    features = check_features(text)
    update_vocab(stemmed)
    vectorized = vectorize_and_pad(stemmed)
    return features + vectorized
