# train.py (Optimized for Spam Detection)

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1️⃣ Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['Category', 'Message']

# 2️⃣ Check class balance
print("Class distribution:\n", df['Category'].value_counts())

# 3️⃣ Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Category'], test_size=0.2, random_state=42, stratify=df['Category']
)

# 4️⃣ TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Train model with balanced class priors
model = MultinomialNB(class_prior=[0.5, 0.5])
model.fit(X_train_tfidf, y_train)

# 6️⃣ Predict on test data
y_pred = model.predict(X_test_tfidf)

# 7️⃣ Evaluation metrics
print("\n Model Evaluation Results:")
print(f" Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f" Precision: {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f" Recall:    {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
print(f" F1 Score:  {f1_score(y_test, y_pred, pos_label='spam'):.4f}")

print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# 8️⃣ Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'], yticklabels=['ham','spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 9️⃣ Save trained objects
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("\n Model and Vectorizer saved successfully!")
