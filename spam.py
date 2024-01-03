import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Function to load and preprocess the dataset
def load_data(file_path):
    """
    Load the dataset and perform basic preprocessing.
    - Remove duplicates
    - Check for missing values
    """
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    return df

# Text cleaning function
def clean_text(text):
    """
    Cleans the text data by removing punctuation and stopwords.
    """
    text = ''.join([char for char in text if char not in string.punctuation])
    return [word for word in text.split() if word.lower() not in stopwords.words('english')]

# Load and preprocess the dataset
df = load_data("/Users/ben/Developer/python/email-spam-detection/emails.csv")
print(df.head())

# Download NLTK stopwords package
nltk.download('stopwords')

# Creating a machine learning pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(analyzer=clean_text)),
    ('classifier', MultinomialNB())
])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size=0.20, random_state=0)

# Train the pipeline
pipeline.fit(x_train, y_train)

# Function to evaluate the model
def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluates the given model on the training and testing datasets.
    Prints out the classification report, confusion matrix, and accuracy.
    """
    for dataset, x, y in [('Training', x_train, y_train), ('Testing', x_test, y_test)]:
        pred = model.predict(x)
        print(f"{dataset} Classification Report:\n{classification_report(y, pred)}")
        print(f"{dataset} Confusion Matrix:\n{confusion_matrix(y, pred)}")
        print(f"{dataset} Accuracy: {accuracy_score(y, pred):.2f}\n")

# Evaluate the model
evaluate_model(pipeline, x_train, y_train, x_test, y_test)