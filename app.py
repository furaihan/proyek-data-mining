import streamlit as st
import pandas as pd
import re
import joblib
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load('model.pkl')  # Ensure you have the correct path to your model

st.title('Indonesian Clickbait Headline Detector')
