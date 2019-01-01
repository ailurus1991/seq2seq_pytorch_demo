#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import random
import math
import os

spacy_de = spacy.load("de")
spacy_en = spacy.load("en")

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)
TGT = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TGT))

print("Number of training examples: {}".format(len(train_data.examples)))
print("Number of validation examples: {}".format(len(valid_data.examples)))
print("Number of testing examples: {}".format(len(test_data.examples)))

print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq=2)
TGT.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 128

device="cuda"

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)
