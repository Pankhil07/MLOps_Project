from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import random
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import random
from  hydra.experimental import initialize, compose
import yaml
import hydra
#raw_dataset = load_dataset(config['dataset_name'],f"{config['source_language']}-{config['target_language']}")
#raw_dataset= load_dataset("opus_books", "en-it")
#metric = load_metric("sacrebleu")

# Define these variables at the module level

# Load configuration from the YAML file
with initialize(config_path="../../conf"):
    cfg = compose(config_name="config.yml")

# Access configuration parameters using OmegaConf
config = OmegaConf.to_container(cfg)
print("Configuration:")
print(yaml.dump(config))
raw_dataset = load_dataset(cfg.dataset_name, f"{cfg.source_language}-{cfg.target_language}")
#raw_dataset = load_dataset(cfg.dataset_name"opus_books", "en-it")
metric = load_metric("sacrebleu")

# Define your split ratios
train_ratio = cfg.train_ratio
validation_ratio = cfg.validation_ratio
test_ratio = cfg.test_ratio

# Calculate the number of examples for each split
all_data = raw_dataset["train"]
total_examples = len(all_data)
used_examples = 500

train_size = int(train_ratio * used_examples)
validation_size = int(validation_ratio * used_examples)
test_size = used_examples - train_size

# Split the dataset
train_set = Dataset.from_dict(all_data[:train_size])
validation_set = Dataset.from_dict(all_data[train_size:train_size + validation_size])
test_set = Dataset.from_dict(all_data[train_size:min(used_examples, total_examples)])

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

# Load the tokenizer using the configuration
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

#__all__ = ["train_set", "validation_set", "test_set", "tokenizer"]

# Load the dataset using the configuration
#raw_dataset = load_dataset(cfg.dataset_name, f"{cfg.source_language}-{cfg.target_language}")
#metric = load_metric("sacrebleu")

# The rest of your code
#raw_dataset = load_dataset(cfg.dataset_name, f"{cfg.source_language}-{cfg.target_language}")
# ...
#tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
# ...




# Get the directory of the current script
#script_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(script_dir)
# Load configuration form YAML file
#config_path = os.path.join(parent_dir, "config.yml")
#with open(config_path, "r") as config_file:
#    config = yaml.safe_load(config_file)


#raw_dataset = load_dataset(config['dataset_name'],f"{config['source_language']}-{config['target_language']}")
#raw_dataset= load_dataset("opus_books", "en-it")
#metric = load_metric("sacrebleu")
#books.save_to_disk('data-en-it')



# Define your split ratios:
#train_ratio = config["train_ratio"]
#validation_ratio = config["validation_ratio"]
#test_ratio = config["test_ratio"]


# Calculate the number of examples for each split
#all_data = raw_dataset["train"]
#total_examples = len(all_data)
#used_examples = 500

#train_size = int(train_ratio * used_examples)
#validation_size = int(validation_ratio * used_examples)
#test_size = used_examples - train_size

# Split the dataset
#train_set = Dataset.from_dict(all_data[:train_size])

#train_set.save_to_disk('train_set')
#validation_set = Dataset.from_dict(all_data[train_size:train_size + validation_size])
#validation_set.save_to_disk('validation_set')
#test_set = Dataset.from_dict(all_data[train_size:min(used_examples, total_examples)])

# Create a DatasetDict to store the splits
#dataset_dict = DatasetDict({
#    "train": train_set,
#    "validation": validation_set,
#    "test": test_set
#})

# Print the number of examples in each split
#print("Training Set:", len(dataset_dict["train"]))
#print("Validation Set:", len(dataset_dict["validation"]))
#print("Testing Set:", len(dataset_dict["test"]))
#print(raw_dataset['train'][0])

# print parent directory
#model_path = os.path.join(os.getcwd(), "opus-mt-en-it")
#print(model_path)
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
#tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

# def tokenize_function(examples):
#   return tokenizer(examples["translation"],padding="max_length", truncation=True)

# tokenized_train = train_set.map(tokenize_function, batched=True)
# tokenized_test = test_set.map(tokenize_function, batched=True)

# tokenized_train.save_to_disk('train_set')
# tokenized_test.save_to_disk('test_set')


# tokenized_train_en = [tokenizer(item["translation"]["en"], padding="max_length", truncation=True) for item in dataset_dict["train"]]
# tokenized_train_it = [tokenizer(item["translation"]["it"], padding="max_length", truncation=True) for item in dataset_dict["train"]]
# tokenized_test_en = [tokenizer(item["translation"]["en"], padding="max_length", truncation=True) for item in dataset_dict["test"]]
# tokenized_test_it = [tokenizer(item["translation"]["it"], padding="max_length", truncation=True) for item in dataset_dict["test"]]
# tokenized_test = test_set.map(tokenize_function, batched=True)
# print(test_set)
#metric