import sys
import os
import wandb
import yaml
import numpy as np
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from hydra.experimental import initialize, compose
from src.models.tokenizer import tokenized_datasets_test,tokenized_datasets_train
from datasets import load_dataset,Dataset,DatasetDict,load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

# Initialize Hydra
#initialize(config_path=".")
metric = load_metric("sacrebleu")
# Load configuration from the YAML file
with initialize(config_path="../../conf"):
    cfg = compose(config_name="config.yml")

# Access configuration parameters using OmegaConf
config = OmegaConf.to_container(cfg)

# Set device (cuda or cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_checkpoint = os.path.join(os.getcwd(), "opus-mt-en-it")
model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(device)

# Initialize WandB with your API key
wandb.login(key="51444d17389535db95f0e83e403f2080038ccfc5")

batch_size = cfg.batch_size
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{cfg.source_language}-to-{cfg.target_language}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=cfg.weight_decay,
    save_total_limit=cfg.save_total_limit,
    num_train_epochs=cfg.num_train_epochs,
    predict_with_generate=True
)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./model.pt")