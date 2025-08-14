import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
# import nltk
from nltk.corpus import words

#! if not download
# nltk.download('words')

word_list = set(words.words())

# Simple text processing function (tokenization and lowercasing)
def simple_text_process(text):
    return text.lower().split()


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """ Fit the Naive Bayes classifier. """
        self.classes = np.unique(y)
        self.feature_count = X.shape[1]
        self.class_log_prior_ = np.log(np.bincount(y) / len(y))
        self.feature_log_prob_ = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            logged_feature_count = np.log(X_cls.sum(axis=0) + self.alpha)
            self.feature_log_prob_[cls] = logged_feature_count - np.log(X_cls.sum() + self.alpha * self.feature_count)

    def predict_proba(self, X):
        """ Predict class probabilities for samples in X. """
        log_probs = self.predict_log_proba(X)
        return np.exp(log_probs) / np.exp(log_probs).sum(axis=1, keepdims=True)

    def predict_log_proba(self, X):
      """ Compute log probability of X for each class. """
      log_probs = np.zeros((X.shape[0], len(self.classes)))
      for idx, cls in enumerate(self.classes):
        log_prob_cls = self.class_log_prior_[cls] + X.dot(self.feature_log_prob_[cls].T)
        if isinstance(log_prob_cls, np.matrix):
            log_prob_cls = log_prob_cls.A  #! Convert matrix to ndarray (left)
        log_probs[:, idx] = log_prob_cls.ravel()
      return log_probs

    def predict(self, X):
      """ Predict class labels for samples in X. """
      log_probs = self.predict_log_proba(X)
      return self.classes[np.argmax(log_probs, axis=1)]



try:
    data = pd.read_csv('data2.csv')

except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

if data.empty:
    print("Error: The data file is empty.")
    exit()

# sentiment encoding
sentiment_mapping = {'regular': 0, 'help': 1, 'complain': 2}
data['Sentiment'] = data['Sentiment'].map(sentiment_mapping)

# text processing
data['Processed_Text'] = data['Text'].apply(simple_text_process)

# Feature Extraction ( TF-IDF )
tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
features = tfidf.fit_transform(data['Processed_Text'])


X_train, X_test, y_train, y_test = train_test_split(features, data['Sentiment'], test_size=0.2, random_state=42)

# Model Training
model = MultinomialNaiveBayes()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)

# Custom Accuracy Calculation
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

def is_legit_text(text):
    words_in_text = simple_text_process(text)
    legit_word_count = sum(word in word_list for word in words_in_text)
    return legit_word_count >= len(words_in_text) * 0.7 # (50%)


# Prediction Function
def predict_sentiment(text, threshold=0.3, min_length=3):
    if len(text) < min_length or not is_legit_text(text):
        return 'invalid'

    processed_text = simple_text_process(text)
    text_features = tfidf.transform([processed_text])
    probabilities = model.predict_proba(text_features)[0]
    max_prob = np.max(probabilities)
    if max_prob < threshold:
        return 'invalid'
    else:
        sentiment = model.predict(text_features)
        sentiment_mapping_inv = {0: 'regular', 1: 'help', 2: 'complain'}
        return sentiment_mapping_inv[sentiment[0]]
    
# Test
input_text = "the door is broken in the entrance"
predicted_sentiment = predict_sentiment(input_text)
print(f"The sentiment of the text '{input_text}' is: {predicted_sentiment}")