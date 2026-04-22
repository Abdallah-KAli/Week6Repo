from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from src.preprocessing import clean_text

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

label_map = {
    0: ("Negative", "😡"),
    1: ("Neutral", "😐"),
    2: ("Positive", "😊"),
    "negative": ("Negative", "😡"),
    "neutral": ("Neutral", "😐"),
    "positive": ("Positive", "😊")
}

def normalize(pred):
    if isinstance(pred, str):
        return label_map.get(pred.lower(), ("Unknown", "❓"))

    try:
        return label_map.get(int(pred), ("Unknown", "❓"))
    except:
        return ("Unknown", "❓")


def get_confidence(text):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        return round(float(np.max(proba)) * 100, 2)
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    text = data.get("text", "")

    cleaned = clean_text(text)

    pred = model.predict([cleaned])[0]

    result, emoji = normalize(pred)

    confidence = get_confidence(cleaned)

    return jsonify({
        "result": result,
        "emoji": emoji,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)