import json
import argparse
import os
import time
from collections import OrderedDict

from tensorflow.keras.models import load_model
from numpy import argsort

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        # load model
        model = load_model(os.path.join(artefacts_path, "model.h5"))

        # TODO: CODE HERE
        # load params
        params = json.load(open(os.path.join(artefacts_path, "params.json")))

        # TODO: CODE HERE
        # load labels_to_index
        labels_to_index = json.load(open(os.path.join(artefacts_path, "labels_index.json")))

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            Predict top_k tags for a list of texts.
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")

        # Generate embeddings for the input text
        embeddings = embed(text_list)

        # Get model predictions
        predictions = self.model.predict(embeddings)

        # Convert predictions to top_k labels
        top_k_predictions = [
            [self.labels_index_inv[idx] for idx in sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:top_k]]
            for pred in predictions
        ]

        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return top_k_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
