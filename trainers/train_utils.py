import csv
import glob
import json
import random
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

from sklearn.metrics import accuracy_score

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.
    
    ########################################################
    # TODO: Please finish the pairwise accuracy computation.
    # Hint: Utilize the `guid` as the `guid` for each
    # statement coming from the same complementary
    # pair is identical. You can simply pair the these
    # predictions and labels w.r.t the `guid`. 
    
    guid_set = set(guids)
    
    for guid in guid_set:
        indices = [i for i, item in enumerate(guids) if item == guid]
        acc = acc + 1 if (np.all(preds[indices] == labels[indices])) else acc
        
    acc = acc / len(guid_set)
        
    # End of TODO
    ########################################################
     
    return acc

def subgroup_accuracies(preds, labels, groups):

    acc = 0.0  # The accuracy to return.
    
    ########################################################
    # TODO: Please finish the pairwise accuracy computation.
    # Hint: Utilize the `guid` as the `guid` for each
    # statement coming from the same complementary
    # pair is identical. You can simply pair the these
    # predictions and labels w.r.t the `guid`. 
    
    group_set = set(groups)
    combined_arr = np.concatenate((np.array([preds]), np.array([labels]), np.array([groups])))
    
    group_accuracies = {}
    
    for group in group_set:
        group_vals = combined_arr[combined_arr[:,2] == group]
        group_preds, group_labels = group_vals[:,0], group_vals[:,1]
        group_accuracies[group] = accuracy_score(group_labels, group_preds)
        
    # End of TODO
    ########################################################
     
    return group_accuracies


if __name__ == "__main__":

    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0,1, 0, 0, 1, 1, 1])
    acc = pairwise_accuracy(guids, preds, labels)
    
    if acc == 0.75:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")
