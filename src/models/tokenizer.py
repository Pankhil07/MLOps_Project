import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from  hydra.experimental import initialize, compose
import src
from src import data
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

#sys.path.append(os.path.join(os.getcwd(), config['src_path']))
#from src.data.data import tokenizer, train_set,test_set,validation_set,raw_dataset
import yaml
from src.data.data import train_set,test_set,validation_set

#tokenizer, train_set, validation_set, test_set = main()
# Get the directory of the current script
#script_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(script_dir)
# Load configuration form YAML file
#config_path = os.path.join(parent_dir, "config.yml")
#with open(config_path, "r") as config_file:
#    config = yaml.safe_load(config_file)

#sys.path.append(os.path.join(os.getcwd(), config['src_path']))
#from src.data.data import  train_set,test_set,validation_set,raw_dataset
# Load configuration from the YAML file
with initialize(config_path="../../conf"):
    cfg = compose(config_name="config.yml")

# Access configuration parameters using OmegaConf
config = OmegaConf.to_container(cfg)
print("Configuration:")
print(yaml.dump(config))
#data_processing_config = cfg.data_processing
#paths_config = cfg.paths

prefix = ""
max_input_length = cfg.max_input_length
max_target_length = cfg.max_target_length
source_language = cfg.source_language
target_language = cfg.target_language
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

def preprocess_function(examples):
    inputs = [prefix + ex[source_language] for ex in examples["translation"]]
    targets = [ex[target_language] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets_train = train_set.map(preprocess_function, batched=True)
print(tokenized_datasets_train)
tokenized_datasets_test = validation_set.map(preprocess_function, batched=True)
print(tokenized_datasets_test)



#prefix = ""
#max_input_length = config["max_input_length"]
#
