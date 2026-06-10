# AI Sentiment Analysis Web App

A web-based sentiment analysis application built using Flask and Machine Learning.

The application allows users to enter text and automatically predicts whether the sentiment is:

* Positive 😊
* Negative 😡
* Neutral 😐

The prediction result is displayed with a confidence score and a modern interactive user interface.

## Features

* Real-time sentiment prediction
* Confidence percentage display
* Clean text preprocessing
* Flask web application
* Modern responsive UI
* Machine Learning classification model
* Support for Positive, Negative, and Neutral sentiment classes

## Technologies Used

  Python
  Flask
  Scikit-learn
  HTML
  CSS
  JavaScript

## Project Structure

├── app.py
├── data/
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── src/
│   └── preprocessing.py
├── templates/
│   └── index.html
└── requirements.txt


## How to Run

Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Open:

http://127.0.0.1:5000

## Example

Input:

I absolutely love this application. It is easy to use and very useful.


Output:

Positive 😊
Confidence: 92%


## License

MIT License


Developed as part of the Optimum Partners Practical Training Program.
