import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm
from .utils import DataProcessor
from .utils import SemEvalSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class SemEvalDataProcessor(DataProcessor):
    """Processor for Sem-Eval 2020 Task 4 Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        ##################################################
        # TODO: Use csv.DictReader or pd.read_csv to load
        # the csv file and process the data properly.
        # We recommend separately storing the correct and
        # the incorrect statements into two individual
        # `examples` using the provided class
        # `SemEvalSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # For the guid, simply use the row number (0-
        # indexed) for each data instance.

        csv_path = os.path.join(data_dir, split+".csv")
        
        df = pd.read_csv(csv_path, delimiter=',')
        
        examples = []
        
        label_is_none = (split == 'test')
        
        for index, row in df.iterrows():
            correct_statement = row['Correct Statement']
            incorrect_statement = row['Incorrect Statement']
            right_reason_1 = row['Right Reason1'] if 'Right Reason1' in row else None
            right_reason_2 = row['Right Reason2'] if 'Right Reason2' in row else None
            right_reason_3 = row['Right Reason3'] if 'Right Reason3' in row else None
            confusing_reason_1 = row['Confusing Reason1'] if 'Confusing Reason1' in row else None
            confusing_reason_2 = row['Confusing Reason2'] if 'Confusing Reason2' in row else None
            
            correct_example = SemEvalSingleSentenceExample(guid=str(index), text=correct_statement, label=1 if not label_is_none else None, right_reason1=right_reason_1, right_reason2=right_reason_2, right_reason3=right_reason_3, confusing_reason1=confusing_reason_1, confusing_reason2=confusing_reason_2)
            incorrect_example = SemEvalSingleSentenceExample(guid=str(index), text=incorrect_statement, label=0 if not label_is_none else None, right_reason1=right_reason_1, right_reason2=right_reason_2, right_reason3=right_reason_3, confusing_reason1=confusing_reason_1, confusing_reason2=confusing_reason_2)
            
            examples.append(correct_example)
            examples.append(incorrect_example)
        
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
    proc = SemEvalDataProcessor(data_dir="datasets/semeval_2020_task4")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
