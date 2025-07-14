import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB  # you can swap with SVM/LogReg
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load JSON
with open("enhanced_intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Extract data
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Step 3: Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Step 5: Build Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 7: Save model and label encoder (optional)
import pickle
with open("zentra_intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("zentra_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)



import joblib
joblib.dump(model, "intent_model.pkl")
