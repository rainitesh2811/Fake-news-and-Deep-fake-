import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
x_train_vec = vectorizer.fit_transform(x_train).toarray()
x_test_vec = vectorizer.transform(x_test).toarray()

# Build Feed-Forward Neural Network model
model = Sequential()
model.add(Dense(512, input_shape=(5000,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_vec, y_train, epochs=10, batch_size=64, validation_data=(x_test_vec, y_test))

# Predict and evaluate the model
y_pred = (model.predict(x_test_vec) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f'Feed-Forward Neural Network Model Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Define manual testing function
def manual_testing(news):
    processed_news = wordopt(news)
    vec = vectorizer.transform([processed_news]).toarray()
    prediction = (model.predict(vec) > 0.5).astype("int32")
    return "Fake News" if prediction[0] == 0 else "True News"

# Input for manual testing
news = input("Enter news text: ")
result = manual_testing(news)
print("The news is classified as:", result)