from datasets import load_dataset,Dataset,DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from sklearn.model_selection import train_test_split

import random
books = load_dataset("opus_books", "en-it")

books.save_to_disk('data-en-it')
print(books)
# Define your split ratios:
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Calculate the number of examples for each split
total_examples = len(books["train"])
train_size = int(train_ratio * total_examples)
validation_size = int(validation_ratio * total_examples)
test_size = total_examples - train_size - validation_size

# Split the dataset
train_set = Dataset.from_dict(books["train"][:train_size])
train_set.save_to_disk('train_set')
validation_set = Dataset.from_dict(books["train"][train_size:train_size + validation_size])
validation_set.save_to_disk('validation_set')
test_set = Dataset.from_dict(books["train"][train_size + validation_size:])
test_set.save_to_disk('test_set')
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


