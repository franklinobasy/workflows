import unittest
import os
import joblib
from project.src.knn_classification import load_iris_data, split_data, train_model, evaluate_model, save_model

class TestIrisClassifier(unittest.TestCase):

    def setUp(self):
        self.iris_df = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.iris_df)
        self.knn_model = train_model(self.X_train, self.y_train)
        self.model_filename = 'test_knn_model.joblib'

    def tearDown(self):
        if os.path.exists(self.model_filename):
            os.remove(self.model_filename)

    def test_model_accuracy(self):
        # Test the accuracy of the trained model on the test set
        accuracy = evaluate_model(self.knn_model, self.X_test, self.y_test)
        self.assertTrue(accuracy >= 0.0, "Accuracy should be a valid value")

    def test_model_saving(self):
        # Save the model to a file
        save_model(self.knn_model, self.model_filename)
        self.assertTrue(os.path.isfile(self.model_filename), f"Model file '{self.model_filename}' not found")

        # Check if the saved model has valid accuracy
        loaded_model = joblib.load(self.model_filename)
        accuracy = evaluate_model(loaded_model, self.X_test, self.y_test)
        self.assertTrue(accuracy >= 0.96, "Accuracy should be a valid value")

if __name__ == '__main__':
    unittest.main()
