from flask import Flask, render_template, request, jsonify
import pickle
from src.preprocessing import clean_text

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

emoji_map = {
    "positive": "😊",
    "negative": "😡",
    "neutral": "😐"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    cleaned = clean_text(text)

    pred = model.predict([cleaned])[0]
    pred_str = str(pred).lower()

    emoji = emoji_map.get(pred_str, "❓")

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        confidence = round(float(max(proba)) * 100, 2)

    return jsonify({
        "result": pred_str,
        "emoji": emoji,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)