import unittest
import pandas as pd
from unittest.mock import MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import utils



class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_test_batches(self):
        """
        Test _get_num_test_batches using mocks for _get_num_test_samples.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_test_samples = MagicMock(return_value=50)
        self.assertEqual(base._get_num_test_batches(), 3)

    def test_get_index_to_label_map(self):
        """
        Test that get_index_to_label_map returns the correct index -> label map.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._label_list = ['label_a', 'label_b', 'label_c']
        expected = {0: 'label_a', 1: 'label_b', 2: 'label_c'}
        self.assertEqual(base.get_index_to_label_map(), expected)

    def test_index_to_label_and_label_to_index_are_identity(self):
        """
        Test that index -> label and label -> index mappings are consistent.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._label_list = ['label_a', 'label_b', 'label_c']
        index_to_label = base.get_index_to_label_map()
        label_to_index = base.get_label_to_index_map()
        for index, label in index_to_label.items():
            self.assertEqual(label_to_index[label], index)

    def test_to_indexes(self):
        """
        Test that to_indexes correctly converts labels to indexes.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._label_list = ['label_a', 'label_b', 'label_c']
        labels = ['label_a', 'label_c', 'label_b']
        expected_indexes = [0, 2, 1]
        self.assertEqual(base.to_indexes(labels), expected_indexes)

class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", min_samples_per_label=1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        print("dataset: \n",dataset)
        print("expected: \n",expected)
        pd.testing.assert_frame_equal(dataset.reset_index(drop=True), expected)
    
    def test__get_num_samples_is_correct(self):
        """
        Test that _get_num_samples returns the correct number of samples.
        """
        # Mock pandas.read_csv pour retourner un DataFrame avec les colonnes attendues
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': range(100),
            'tag_name': ['tag'] * 100,
            'tag_id': range(100),
            'tag_position': [0] * 100,
            'title': ['title'] * 100
        }))

        # Créez l'objet LocalTextCategorizationDataset
        dataset = utils.LocalTextCategorizationDataset("fake_path", 20)

        # Vérifiez que _get_num_samples retourne le nombre correct d'échantillons
        self.assertEqual(dataset._get_num_samples(), 100)


    def test_get_train_batch_returns_expected_shape(self):
        """
        Test that get_train_batch returns the expected shape using MagicMock.
        """
        # Créez une instance simulée de LocalTextCategorizationDataset
        dataset = MagicMock(spec=utils.LocalTextCategorizationDataset)

        # Simulez les attributs nécessaires
        dataset.batch_size = 20
        dataset.x_train = pd.Series(range(100))
        dataset.y_train = pd.Series(range(100))
        dataset.train_batch_index = 0
        dataset.preprocess_text = MagicMock(side_effect=lambda x: x)  

        dataset._get_num_train_batches = MagicMock(return_value=5) 


    def test_get_test_batch_returns_expected_shape(self):
        """
        Test that get_train_batch returns the expected shape using MagicMock.
        """
        # Créez une instance simulée de LocalTextCategorizationDataset
        dataset = MagicMock(spec=utils.LocalTextCategorizationDataset)

        # Simulez les attributs nécessaires
        dataset.batch_size = 20
        dataset.x_test = pd.Series(range(100))
        dataset.y_test = pd.Series(range(100))
        dataset.test_batch_index = 0
        dataset.preprocess_text = MagicMock(side_effect=lambda x: x)  

        dataset._get_num_test_batches = MagicMock(return_value=5) 

    def test_get_train_batch_raises_assertion_error(self):
        """
        Test that get_train_batch raises an AssertionError when no train samples are available.
        """
        dataset = utils.LocalTextCategorizationDataset("fake_path", 20)
        dataset._get_num_train_batches = MagicMock(return_value=0)
        with self.assertRaises(AssertionError):
            dataset.get_train_batch()
