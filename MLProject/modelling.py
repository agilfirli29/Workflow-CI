import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("fake_news_preprocessing.csv")

# Cleaning
df.dropna(subset=["text", "label"], inplace=True)
df["text"] = df["text"].astype(str)
df.drop_duplicates(inplace=True)

print("Jumlah data setelah cleaning:", df.shape)

# Features
X = df["text"]
y = df["label"]

# TFIDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7
)

X_tfidf = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(
    C=10,
    max_iter=100
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# MLflow Logging
mlflow.log_param("model_type", "Logistic Regression")
mlflow.log_param("C", 10)
mlflow.log_param("max_iter", 100)

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

# Classification report
report = classification_report(y_test, y_pred)

with open("classification_report.txt", "w") as f:
    f.write(report)

mlflow.log_artifact("classification_report.txt")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Fake", "Real"])
plt.yticks([0, 1], ["Fake", "Real"])

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='black'
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

mlflow.log_artifact("confusion_matrix.png")

# Save model
mlflow.sklearn.log_model(model, "model")

print("\n===== EVALUATION =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nModel berhasil disimpan ke MLflow!")