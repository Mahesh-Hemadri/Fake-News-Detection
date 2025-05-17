import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and prepare data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
data["text"] = data["title"] + " " + data["text"]

# Print dataset distribution
print("Dataset Distribution:")
print(f"Number of fake news articles: {len(fake)}")
print(f"Number of real news articles: {len(true)}")
print(f"Total articles: {len(data)}\n")


# Visualize class distribution
category_counts = data['label'].value_counts()
category_names = ['Fake', 'Real']

plt.figure(figsize=(6,4))
plt.bar(category_names, category_counts, color=['red', 'green'])
plt.title('Distribution of News Categories')
plt.xlabel('Category')
plt.ylabel('Number of Articles')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Seaborn count plot
sns.countplot(x=data['label'])
plt.title("Count Plot of News Categories")
plt.xlabel("Label (0 = Fake, 1 = Real)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Train/Test split
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

# Classification report
report = classification_report(y_test, preds, target_names=["Fake", "Real"])
print(report)

# Save model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
