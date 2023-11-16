from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import joblib
import os

def load_iris_data():
    # Load the iris dataset
    iris = datasets.load_iris()

    # Create a DataFrame from the iris dataset
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                            columns=iris['feature_names'] + ['target'])

    return iris_df

def split_data(df):
    # Split the data into features (X) and target variable (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_neighbors=3):
    # Initialize the k-nearest neighbor model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model to the training data
    knn_model.fit(X_train, y_train)

    return knn_model

def evaluate_model(model: KNeighborsClassifier, X_test, y_test):
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, model_filename):
    # Save the trained model to a file
    joblib.dump(model, model_filename)
    print(f'Model saved as {model_filename}')

def main():
    # Load data
    iris_df = load_iris_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(iris_df)

    # Train the model
    knn_model = train_model(X_train, y_train)

    # Evaluate and print accuracy (optional)
    evaluate_model(knn_model, X_test, y_test)

    # Save the model
    model_filename = 'project/model/knn_model.joblib'
    save_model(knn_model, model_filename)

if __name__ == "__main__":
    main()
