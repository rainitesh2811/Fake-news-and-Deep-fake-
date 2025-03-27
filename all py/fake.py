import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
data_fake = pd.read_csv('c:/Users/Mantoo/Desktop/Projects/all py/all py/Fake.csv')
data_true = pd.read_csv('c:/Users/Mantoo/Desktop/Projects/all py/all py/True.csv')

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

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test_vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the neural network model
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(5000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

model = FeedForwardNN()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Predict and evaluate the model
with torch.no_grad():
    y_pred = model(x_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, y_pred)
    print(f'Feed-Forward Neural Network Model Accuracy: {accuracy}')
    print(classification_report(y_test_tensor, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test_tensor, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Define manual testing function
def manual_testing(news):
    processed_news = wordopt(news)
    vec = vectorizer.transform([processed_news]).toarray()
    vec_tensor = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(vec_tensor)
        prediction = (prediction > 0.5).float()
    return "Fake News" if prediction[0] == 0 else "True News"

# Input for manual testing
news = input("Enter news text: ")
result = manual_testing(news)
print("The news is classified as:", result)