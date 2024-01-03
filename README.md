# Email Spam Detection with Machine Learning

## Project Overview

This project implements a machine learning model to classify emails into 'spam' or 'non-spam' categories. Utilizing Python and NLP techniques, the model analyzes the content of emails to detect spam. This repo contains the full code and datasets used in the project.

## Technologies Used

- Python
- NumPy
- Pandas
- NLTK
- scikit-learn

## Installation

First, clone this repository to your local machine using:

```
git clone https://github.com/blakeben/email-spam-detection-ml
```

After cloning, install the required packages using:

```
pip install -r requirements.txt
```

## How to Run

Navigate to the project directory and run the script using:

```
python email_spam_detector.py
```

## Features

- Data preprocessing: Cleaning email text data and removing stopwords.
- Text vectorization using CountVectorizer.
- Training a Naive Bayes classifier to identify spam emails.
- Evaluating the model's performance on unseen data.

## Dataset

The dataset used in this project contains various email texts labeled as 'spam' or 'non-spam'.

## Author

- Ben Blake - benblake0@outlook.com
