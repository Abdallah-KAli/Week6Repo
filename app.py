from flask import Flask, render_template, request
from src.predict import predict

app = Flask(__name__)

label_map = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("text")

        if text:
            pred = predict(text)

            print("RAW PRED:", pred)

            if isinstance(pred, str):
                result = label_map.get(pred.lower(), "Unknown")
            else:
                result = "Unknown"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)