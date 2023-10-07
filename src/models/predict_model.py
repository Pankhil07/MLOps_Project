from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import os
from src.data.data import test_set
import csv
# Load the fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")


#english_sentences = [item['text'] for item in test_data if item['language'] == 'en']
# Define a translation pipeline
translator = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-it",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Use the test_set directly
test_data = test_set

english_sentences = []
for i in test_data:
    english_sentences.append(i['translation'].get('en'))


#print(type(english_sentences))

translations_list = []

# Iterate through the test data and generate translations
for input_text in english_sentences:
    # Generate translations
    translations = translator(input_text, max_length=400)  # You can adjust the max_length as needed
    translations_list.append(translations[0]["translation_text"])
    # Display the translations
   # for translation in translations:
        #print("Input: ", input_text)
        #print("Translation: ", translation["translation_text"])
        #print()



csv_filename = "translations_new.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["Input Sentence", "Translation"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write the data rows
    for input_text, translation in zip(english_sentences, translations_list):
        writer.writerow({"Input Sentence": input_text, "Translation": translation})

print(f"Translations saved to {csv_filename}")