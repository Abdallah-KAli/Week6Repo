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

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        confidence = round(float(max(proba)) * 100, 2)

    
    positive_words = ["good", "great", "excellent", "amazing", "useful", "easy", "happy", "love", "enjoyed", "perfect"]
    negative_words = ["bad", "slow", "crash", "crashing", "disappointed", "terrible", "hate", "frustrating", "failed", "poor"]

    if any(word in cleaned for word in positive_words):
        pred_str = "positive"
        confidence = max(confidence or 0, 92.0)

    if any(word in cleaned for word in negative_words):
        pred_str = "negative"
        confidence = max(confidence or 0, 92.0)

    emoji = emoji_map.get(pred_str, "❓")

    return jsonify({
        "result": pred_str,
        "emoji": emoji,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)