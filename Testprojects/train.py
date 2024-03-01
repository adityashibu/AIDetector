import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# # Model using SVC
# def train_svm_model(data_file):
#     # Load preprocessed data
#     data = pd.read_csv(data_file)

#     # Extract feature labels
#     X = data['filtered_text']
#     y = data['generated']

#     # Split data into training and testing data sets
#     X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

#     # Convert text data into numerical features using CountVectorizer
#     vectorizer = CountVectorizer()
#     X_train_vectorized = vectorizer.fit_transform(X_train)
#     X_val_vectorized = vectorizer.transform(X_val)
    
#     # Initialize and train a logistic regression model
#     logreg = LogisticRegression(max_iter=1000)
#     logreg.fit(X_train_vectorized, y_train)

#     # Evaluate the model on the validation set
#     y_val_pred = svm.predict(X_val_vectorized)
#     accuracy = accuracy_score(y_val, y_val_pred)
#     print("Validation accuracy: ", accuracy)
#     print("Classification Report: ")
#     print(classification_report(y_val, y_val_pred))

#     return svm, vectorizer  # Return the trained model and vectorizer

# if __name__ == "__main__":
#     trained_model, vectorizer = train_svm_model("preprocessed_data.csv")

# Model using Logistic Regression
def train_logistic_regression_model(data_file):
    # Load preprocessed data
    data = pd.read_csv(data_file)

    # Extract feature labels
    X = data['filtered_text']
    y = data['generated']

    # Split data into training and testing data sets
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Convert text data into numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_val_vectorized = vectorizer.transform(X_val_test)

    # Initialize and train a logistic regression model
    logreg = LogisticRegression(max_iter=100, penalty='l2')
    logreg.fit(X_train_vectorized, y_train)
    
    scores = cross_val_score(logreg, X_train_vectorized, y_train, cv=5)

    # Evaluate the model on the validation set
    y_val_pred = logreg.predict(X_val_vectorized)
    accuracy = accuracy_score(y_val_test, y_val_pred)
    print("Validation accuracy: ", accuracy)
    print("Classification Report: ")
    print(classification_report(y_val_test, y_val_pred))
    
    print(scores)

    return logreg, vectorizer  # Return the trained model and vectorizer

if __name__ == "__main__":
   trained_model, vectorizer = train_logistic_regression_model("preprocessed_data.csv")

