Multinomial Naive Bayes Text Classifier

This project demonstrates a text classification system using the Multinomial Naive Bayes algorithm. It focuses on analyzing short text messages and categorizing them into predefined sentiment classes such as regular, help, and complain.

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features that represent the importance of each word relative to the dataset. Before vectorization, the text is preprocessed by tokenizing and converting to lowercase.

A custom Naive Bayes classifier is implemented from scratch, calculating class priors and feature probabilities without relying on external machine learning libraries. The model also includes a text validity check using an English word list to filter out nonsensical input.

This approach illustrates the principles of probabilistic text classification, feature extraction, and basic natural language preprocessing, providing a foundation for more advanced sentiment analysis systems.
