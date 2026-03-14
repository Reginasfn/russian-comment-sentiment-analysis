import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model as keras_load_model
import warnings
import os

warnings.filterwarnings('ignore')

MODEL_PATH = 'models/best_model_streamlit.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'

MAX_LEN = 24
CLASS_NAMES = ['нормальный', 'оскорбление', 'угроза', 'непристойность']

_model = None
_tokenizer = None


def get_model():
    global _model
    if _model is None:
        print(f"📥 Загрузка модели: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

        _model = keras_load_model(MODEL_PATH)

        print("Модель загружена!")
    return _model


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print(f"Загрузка токенизатора: {TOKENIZER_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Токенизатор не найден: {TOKENIZER_PATH}")
        with open(TOKENIZER_PATH, 'rb') as f:
            _tokenizer = pickle.load(f)
        print(f"Токенизатор загружен! Слов: {len(_tokenizer.word_index)}")
    return _tokenizer


def predict_toxicity(text):
    model = get_model()
    tokenizer = load_tokenizer()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    prediction = model.predict(padded, verbose=0)[0]

    predicted_idx = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_idx]

    toxic_classes = []
    for i, class_name in enumerate(CLASS_NAMES):
        if class_name != 'нормальный' and prediction[i] > 0.5:
            toxic_classes.append(class_name)

    is_toxic = len(toxic_classes) > 0

    return {
        'probabilities': prediction,
        'predicted_class': predicted_class,
        'toxic_classes': toxic_classes,
        'is_toxic': is_toxic
    }