from data import dataset_dict
from datasets import load_dataset,Dataset,DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
#from sklearn.model_selection import train_test_split

train_en =  [item["translation"]["en"] for item in dataset_dict["train"]]

train_it = [item["translation"]["it"] for item in dataset_dict["train"]]

test_en =  [item["translation"]["en"] for item in dataset_dict["test"]]
test_it = [item["translation"]["it"] for item in dataset_dict["test"]]


