import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset
data = {
    "text": [
        "I am so happy today!",
        "This is so sad and depressing.",
        "I am very angry with you!",
        "I'm scared of the dark.",
        "What a surprise, I didn’t expect this!",
        "I feel calm and relaxed now."
    ],
    "emotion": ["happy", "sad", "angry", "fear", "surprise", "neutral"]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")
