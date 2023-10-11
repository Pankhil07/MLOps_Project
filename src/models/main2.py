import sys
import os
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
sys.path.append(os.path.join(os.getcwd(), "src"))
from datasets import load_dataset,Dataset,DatasetDict,load_metric
from src.models.tokenizer import tokenized_datasets_train,tokenized_datasets_test,tokenizer
import torch
import wandb
import yaml
# Get the directory of the current script

# Load configuration form YAML file
config_path = os.path.join('.', "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

metric = load_metric("sacrebleu")



model_checkpoint = os.path.join(os.getcwd(), "opus-mt-en-it")
# model_checkpoint ="/home/pankhil/PycharmProjects/MLOps/opus-mt-en-it"
model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"]).to(device)



# Initialize WandB with your API key
wandb.login(key="51444d17389535db95f0e83e403f2080038ccfc5")
#"51444d17389535db95f0e83e403f2080038ccfc5"
batch_size = config["batch_size"]
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    report_to = ['wandb'],
    output_dir = f"{model_name}-finetuned-{config['source_language']}-to-{config['target_language']}",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    max_steps = 5,
    logging_steps = 5,
    eval_steps = 5,
    save_steps = 5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=config["weight_decay"],
    save_total_limit=config["save_total_limit"],
    num_train_epochs=config["num_train_epochs"],
    predict_with_generate=True,
    load_best_model_at_end = True,
    metric_for_best_model = 'accuracy',
    run_name = 'custom_training',
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
import numpy as np
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"accuracy": result["score"]}
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
    compute_metrics=compute_metrics,

)
trainer.evaluate()
trainer.train()

model.save_pretrained("./model.pt")
wandb.finish()