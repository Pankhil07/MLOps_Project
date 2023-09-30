from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from sklearn.model_selection import train_test_split

import random
raw_dataset= load_dataset("opus_books", "en-it")
metric = load_metric("sacrebleu")
#books.save_to_disk('data-en-it')



# Define your split ratios:
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Calculate the number of examples for each split
total_examples = len(raw_dataset["train"])
train_size = int(train_ratio * total_examples)
validation_size = int(validation_ratio * total_examples)
test_size = total_examples - train_size

# Split the dataset
train_set = Dataset.from_dict(raw_dataset["train"][:train_size])

#train_set.save_to_disk('train_set')
validation_set = Dataset.from_dict(raw_dataset["train"][train_size:train_size + validation_size])
#validation_set.save_to_disk('validation_set')
test_set = Dataset.from_dict(raw_dataset["train"][train_size:])

# Create a DatasetDict to store the splits
dataset_dict = DatasetDict({
    "train": train_set,
    "validation": validation_set,
    "test": test_set
})

# Print the number of examples in each split
print("Training Set:", len(dataset_dict["train"]))
print("Validation Set:", len(dataset_dict["validation"]))
print("Testing Set:", len(dataset_dict["test"]))

print(raw_dataset['train'][0])
tokenizer = AutoTokenizer.from_pretrained("/home/pankhil/PycharmProjects/MLOps/opus-mt-en-it")
#def tokenize_function(examples):
#   return tokenizer(examples["translation"],padding="max_length", truncation=True)
#tokenized_train = train_set.map(tokenize_function, batched=True)
#tokenized_test = test_set.map(tokenize_function, batched=True)

#tokenized_train.save_to_disk('train_set')
#tokenized_test.save_to_disk('test_set')


#tokenized_train_en = [tokenizer(item["translation"]["en"], padding="max_length", truncation=True) for item in dataset_dict["train"]]
#tokenized_train_it = [tokenizer(item["translation"]["it"], padding="max_length", truncation=True) for item in dataset_dict["train"]]
#tokenized_test_en = [tokenizer(item["translation"]["en"], padding="max_length", truncation=True) for item in dataset_dict["test"]]
#tokenized_test_it = [tokenizer(item["translation"]["it"], padding="max_length", truncation=True) for item in dataset_dict["test"]]
#tokenized_test = test_set.map(tokenize_function, batched=True)
#print(test_set)
metric