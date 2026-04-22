import pickle
from src.preprocessing import clean_text

MODEL_PATH = "models/model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict(text):
    text = clean_text(text)
    return model.predict([text])[0]