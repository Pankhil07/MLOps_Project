from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from sklearn.model_selection import train_test_split
#from  hydra.experimental import initialize, compose
import src
from src import data
#import hydra
from src.data.data import split_data
from omegaconf import DictConfig, OmegaConf
import yaml

# Load configuration from the YAML file
config_path = os.path.join('..\\.', "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)
# data_processing_config = cfg.data_processing
# paths_config = cfg.paths
def test_load_data():
    # Test loading the dataset
    dataset = load_dataset("opus_books", "en-it")
    assert len(dataset) > 0  # Check if the dataset contains data

def test_split_dataset():
    #Test splitting dataset
    # data_processing_config = cfg.data_processing
    # paths_config = cfg.paths

    raw_dataset = load_dataset("opus_books", "en-it")
    train_set, validation_set, test_set = split_data(raw_dataset,config)

    assert len(train_set) > 0  # Check if the training set is not empty
    assert len(validation_set) > 0  # Check if the validation set is not empty
    assert len(test_set) > 0  # Check if the test set is not empty