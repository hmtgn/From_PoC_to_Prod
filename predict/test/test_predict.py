import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from predict.predict.run import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):
    def setUp(self):
        # Mock the model, params, and labels_to_index
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [[0.1, 0.9, 0.8]]
        self.params = {}
        self.labels_to_index = {"tag1": 0, "tag2": 1, "tag3": 2}
        self.model = TextPredictionModel(self.mock_model, self.params, self.labels_to_index)

    @patch("predict.predict.run.embed")  # Updated patch path
    def test_predict(self, mock_embed):
        # Mock the embed function
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        # Test inputs
        text_list = ["Example text"]

        # Call predict
        predictions = self.model.predict(text_list, top_k=2)

        # Print predictions for debugging
        print("Predictions:", predictions)

        # Assertions
        self.mock_model.predict.assert_called_once()
        mock_embed.assert_called_once_with(text_list)
        self.assertEqual(predictions, [["tag2", "tag3"]])

if __name__ == "__main__":
    unittest.main()
