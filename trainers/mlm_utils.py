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

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    inputs should be tokenized token ids with size: (batch size X input length).
    """

    # The eventual labels will have the same size of the inputs,
    # with the masked parts the same as the input ids but the rest as
    # args.mlm_ignore_index, so that the cross entropy loss will ignore it.
    labels = inputs.clone()

    # Constructs the special token masks.
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    ##################################################
    # Optional TODO: this is an optional TODO that can get you more familiarized
    # with masked language modeling.
    
    # First sample a few tokens in each sequence for the MLM, with probability
    # `args.mlm_probability`.
    # Hint: you may find these functions handy: `torch.full`, Tensor's built-in
    # function `masked_fill_`, and `torch.bernoulli`.
    # Check the inputs to the bernoulli function and use other hinted functions
    # to construct such inputs.
    
    labels_device, inputs_device = labels.device, inputs.device
    
    # if labels_device != 'cpu':
    #     labels = labels.to('cpu')
    # if inputs_device != 'cpu':
    #     inputs = inputs.to('cpu')
    
    prob_matrix = torch.full(inputs.size(), args.mlm_probability)
    reduced_prob_matrix = prob_matrix.masked_fill(special_tokens_mask, 0)
    remaining_tokens = torch.bernoulli(reduced_prob_matrix)
    
    if labels_device != 'cpu':
        labels = labels.masked_fill(remaining_tokens.to(labels_device) == 0, 0)
    else:
        labels = labels.masked_fill(remaining_tokens == 0, 0)

    # Remember that the "non-masked" parts should be filled with ignore index.
    
    if labels_device != 'cpu':
        labels = labels.masked_fill(remaining_tokens.to(labels_device) == 0, args.mlm_ignore_index)
    else:
        labels = labels.masked_fill(remaining_tokens == 0, args.mlm_ignore_index)
    
    # For 80% of the time, we will replace masked input tokens with  the
    # tokenizer.mask_token (e.g. for BERT it is [MASK] for for RoBERTa it is
    # <mask>, check tokenizer documentation for more details)
    
    chosen_eighty_percent_of__masked_tokens = torch.bernoulli(torch.full(inputs.size(), 0.8).masked_fill(remaining_tokens != 1, 0))
    if inputs_device:
        inputs = inputs.masked_fill(chosen_eighty_percent_of__masked_tokens.to(inputs_device) == 1, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    else:
        inputs = inputs.masked_fill(chosen_eighty_percent_of__masked_tokens == 1, tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    remaining_tokens = remaining_tokens.masked_fill(chosen_eighty_percent_of__masked_tokens == 1, 0)

    # For 10% of the time, we replace masked input tokens with random word.
    # Hint: you may find function `torch.randint` handy.
    # Hint: make sure that the random word replaced positions are not overlapping
    # with those of the masked positions, i.e. "~indices_replaced".
    
    chosen_ten_percent_of__masked_tokens = torch.bernoulli(torch.full(inputs.size(), 0.1).masked_fill(remaining_tokens != 1, 0))
    if inputs_device != 'cpu':
        inputs = inputs.masked_fill(chosen_ten_percent_of__masked_tokens.to(inputs_device) == 1, 0) + torch.randint(0, len(tokenizer), inputs.size()).masked_fill(chosen_ten_percent_of__masked_tokens != 1, 0).to(inputs_device)
    else:
        inputs = inputs.masked_fill(chosen_ten_percent_of__masked_tokens == 1, 0) + torch.randint(0, len(tokenizer), inputs.size()).masked_fill(chosen_ten_percent_of__masked_tokens != 1, 0)
    
    # if labels_device != 'cpu':
    #     labels = labels.to(labels_device)
    # if inputs_device != 'cpu':
    #     inputs = inputs.to(inputs_device)
    # End of TODO
    ##################################################

    # For the rest of the time (10% of the time) we will keep the masked input
    # tokens unchanged
    pass  # Do nothing.

    return inputs, labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


if __name__ == "__main__":

    class mlm_args(object):
        def __init__(self):
            self.mlm_probability = 0.4
            self.mlm_ignore_index = -100
            self.device = "cpu"
            self.seed = 42
            self.n_gpu = 0

    args = mlm_args()
    set_seed(args)

    # Unit-testing the MLM function.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_sentence = "I am a good student and I love NLP."
    input_ids = tokenizer.encode(input_sentence)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    
    print('input_ids:', input_ids)
    print('input_labels:', input_ids)
    
    inputs, labels = mask_tokens(input_ids, tokenizer, args,
                                 special_tokens_mask=None)
    inputs, labels = list(inputs.numpy()[0]), list(labels.numpy()[0])
    ans_inputs = [101, 146, 103, 170, 103, 2377, 103, 146, 1567, 103, 2101, 119, 102]
    ans_labels = [-100, -100, 1821, -100, 1363, -100, 1105, -100, -100, 21239, -100, -100, -100]
    
    print('inputs:', inputs)
    print('labels:', labels)
    
    if inputs == ans_inputs and labels == ans_labels:
        print("Your `mask_tokens` function is correct!")
    else:
        raise NotImplementedError("Your `mask_tokens` function is INCORRECT!")
