import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load data
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Label data
data_fake["class"] = 0
data_true["class"] = 1

# Create manual testing sets
data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

# Save manual testing sets to CSV
data_fake_manual_testing.to_csv('Fake_manual_testing.csv', index=False)
data_true_manual_testing.to_csv('True_manual_testing.csv', index=False)

# Merge data
data_merge = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data_merge.drop(["title", "subject", "date"], axis=1)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Preprocess text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

# Split data
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Vectorize text
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Performance:")
print(classification_report(y_test, pred_lr))

# Decision Tree Classifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Performance:")
print(classification_report(y_test, pred_dt))

# Gradient Boosting Classifier
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
print("Gradient Boosting Performance:")
print(classification_report(y_test, pred_gb))

# Random Forest Classifier
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
print("Random Forest Performance:")
print(classification_report(y_test, pred_rf))

# Define manual testing function
def manual_testing(news):
    processed_news = wordopt(news)
    vectorized_input = vectorizer.transform([processed_news])
    prediction = RF.predict(vectorized_input)  # Using Random Forest as an example
    return "Fake News" if prediction[0] == 0 else "True News"

# Input for manual testing
news = input("Enter news text: ")
result = manual_testing(news)
print("The news is classified as:", result)