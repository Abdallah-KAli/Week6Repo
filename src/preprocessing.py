import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
important_words = {"not", "no", "very", "too", "but", "never", "really", "so"}

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [w for w in words if (w not in stop_words) or (w in important_words)]

    return " ".join(words)