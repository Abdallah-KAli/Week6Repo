import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download(
    "crowdflower/twitter-airline-sentiment"
)

print("Dataset downloaded at:", path)

file_path = os.path.join(path, "Tweets.csv")

df = pd.read_csv(file_path)

df = df[["text", "airline_sentiment"]]
df.columns = ["text", "label"]

df.to_csv("data/data.csv", index=False)

print("Saved to data/data.csv")