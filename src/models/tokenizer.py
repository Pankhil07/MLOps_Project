import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from data.data import tokenizer, train_set,test_set,validation_set,raw_dataset

prefix = ""
max_input_length = 128
max_target_length  = 128
source_lang = "en"
target_lang = "it"

def preprocess_function(examples):
    inputs = [prefix+ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang]for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets_train = train_set.map(preprocess_function, batched=True)
print(tokenized_datasets_train)
tokenized_datasets_test = validation_set.map(preprocess_function,batched = True)
print(tokenized_datasets_test)

