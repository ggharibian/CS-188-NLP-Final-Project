import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}
        self.str2bool = {"True": True, "False": False}
        self.domain2int = {"physical": 1, "social": 2, "time": 3}
        self.scenario2int = {"comparison": 1, "causal": 2}
        self.numeracy2int = self.label2int

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        ##################################################
        # TODO: Use json python package to load the data
        # properly.
        # We recommend separately storing the two
        # complementary statements into two individual
        # `examples` using the provided class
        # `Coms2SenseSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # Make sure to add to the examples strictly
        # following the `_1` and `_2` order, that is,
        # `sent_1`'s info should go in first and then
        # followed by `sent_2`'s, otherwise your test
        # results will be messed up!
        # For the guid, simply use the row number (0-
        # indexed) for each data instance, i.e. the index
        # in a for loop. Use the same guid for statements
        # coming from the same complementary pair.
        # Make sure to handle if data do not have
        # labels field.
        
        
        if data_dir is None:
            data_dir = self.data_dir

        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        
        examples = []

        for i in range(len(data)):
            example_pair = data[i]
            
            sent_1 = example_pair['sent_1']
            sent_2 = example_pair['sent_2']
            label_1 = self.label2int[example_pair['label_1']] if 'label_1' in example_pair else None
            label_2 = self.label2int[example_pair['label_2']] if 'label_2' in example_pair else None
            domain = self.domain2int[example_pair['domain']] if 'domain' in example_pair else None
            scenario = self.scenario2int[example_pair['scenario']] if 'scenario' in example_pair else None
            numeracy = self.numeracy2int[example_pair['numeracy']] if 'numeracy' in example_pair else None
            
            example_1 = Coms2SenseSingleSentenceExample(str(i), sent_1, label_1, domain, scenario, numeracy)
            example_2 = Coms2SenseSingleSentenceExample(str(i), sent_2, label_2, domain, scenario, numeracy)
            
            examples.append(example_1)
            examples.append(example_2)
        
        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
