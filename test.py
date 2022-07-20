from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model # Add 'load_model'
from joblib import dump, load # For reading the Tokenizer Pickle

KERAS_MODEL = "model.h5"
TOKENIZER_MODEL = "tokenizer.pkl"

# KERAS
SEQUENCE_LENGTH = 30

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# Load the model and the tokenizer to make predictions
model = load_model(KERAS_MODEL)
tokenizer = load(TOKENIZER_MODEL)

def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = "NEUTRAL"
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = "NEGATIVE"
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = "POSITIVE"
        return label
    else:
        return "NEGATIVE" if score < 0.5 else "POSITIVE"


def predict(text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}

print(predict("today i felt great joy in killing a person"))