import streamlit as st
import pandas as pd
import re
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json

# Load the model
model = joblib.load('model.pkl')
stemmer = StemmerFactory().create_stemmer()
angka = ['nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan']
potential_clickbait_words = ['bikin', 'viral', 'gara', 'fakta', 'kejut']
padding_length = 27
with open('vocab.json', 'r') as vocab_file:
    vocab = json.load(vocab_file)

def preprocess_text(text):
    # remove non-alphabet characters except exclamations and question marks and keep numbers
    text = re.sub(r'[^A-Za-z0-9?!]', ' ', text)
    # remove whitespace
    text = text.strip()
    # remove newline
    text = text.replace('\n', ' ')
    # remove extra space
    text = re.sub(' +', ' ', text)
    # lowercase
    text = text.lower()
    # tokenize
    text = word_tokenize(text)
    result = text;
    return result


def check_features(text):
    result = []
    # check if the text contains exclamation mark
    result.append(1 if '!' in text else 2)
    # check if the text contains question mark
    result.append(3 if '?' in text else 4)
    # check if the text contains multiple exclamation marks
    result.append(5 if '!!' in text else 6)
    # check if the text contains digit
    result.append(7 if any(char.isdigit() for char in text) else 8)
    # check if the text contains numbers
    result.append(9 if any(word.lower() in angka for word in text.split()) else 10)
    # check if the text contains potential clickbait words
    result.append(11 if any(word.lower() in potential_clickbait_words for word in text.split()) else 12)
    return result

def update_vocab(text):
    global vocab
    last_index = len(vocab) + 1000
    for word in text:
        if word not in vocab:
            vocab[word] = last_index
            last_index += 1  # Increment last_index for the next new word


st.title('Indonesian Clickbait Headline Detector')

user_input = st.text_input("Masukkan Judul Berita:")

def process_input(text):
    df_one = pd.DataFrame([text], columns=['title'])
    df_one['preprocessed'] = df_one['title'].apply(preprocess_text)
    df_one['stemmed'] = df_one['preprocessed'].apply(lambda x: [stemmer.stem(word) for word in x])
    df_one['features'] = df_one['preprocessed'].apply(lambda x: check_features(' '.join(x)))
    update_vocab(df_one['stemmed'].values[0])
    df_one['vectorized'] = df_one['stemmed'].apply(lambda x: [vocab[word] for word in x if word in vocab])
    df_one['padded'] = df_one['vectorized'].apply(lambda x: (x[:padding_length] if len(x) > padding_length else x + [-1] * (padding_length - len(x))))
    df_one['padded'] = df_one.apply(lambda x: x['features'] + x['padded'], axis=1)
    return df_one

if st.button('Preprocess'):
    df_one = process_input(user_input)
    st.write(df_one)

if st.button('Predict'):
    df_one = process_input(user_input)
    prediction = model.predict(df_one['padded'].values.tolist())[0]
    if prediction == 0:
        st.write('Headline ini bukan clickbait')
    else:
        st.write('Headline ini clickbait')

