import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# A simple text cleaning function (adjust according to your needs)
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Function to preprocess the dataset
def preprocess_data(data):
    # Apply the cleaning function to the review column
    data['cleaned_review'] = data['review'].apply(clean_text)
    # Convert labels to binary (adjust this if your labels are different)
    data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    return data['cleaned_review'], data['label']

# Function to split the dataset
def split_dataset(features, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.5, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to count and print the distribution of the dataset
def print_distribution(y_train, y_val, y_test):
    for dataset, name in zip([y_train, y_val, y_test], ['Training', 'Validation', 'Test']):
        counter = Counter(dataset)
        total = len(dataset)
        print(f"{name} set: Total = {total}, Positive = {counter[1]}, Negative = {counter[0]}")
        print(f"{name} set class distribution: Positive = {(counter[1] / total) * 100:.2f}%, Negative = {(counter[0] / total) * 100:.2f}%\n")

# Load dataset
file_path = 'IMDB Dataset.csv'  # Replace with your dataset path
data = load_dataset(file_path)

# Preprocess data
features, labels = preprocess_data(data)

# Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(features, labels)

# Print the distribution of the dataset
print_distribution(y_train, y_val, y_test)
