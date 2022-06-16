import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from datasets import load_dataset, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

import re
import csv

# Set seed before initializing model.
set_seed(32)

# Choose LM
model_name = 'gpt2'
model_path = 'whitenoise'
    
# Pull up novels
titles = ['Americana',
          'Cosmopolis',
          'End Zone',
          'Great Jones Street',
          'Libra',
          'Mao II',
          'Players',
          'Ratners Star',
          'Running Dog',
          'The Names',
          'Underworld',
          'White Noise',
          'Zero K']


#titles = ['Americana']

path = '/home/danieljhreuter/texts'
#path = 'C:/Users/Daniel/Documents/projects/whitenoise/texts'

# Preprocess texts:
# 1) cut empty lines
# 2) cut lines that indicate the start of a new chapter
for title in titles:
    with open(f"{path}/{title}.txt","r", encoding='utf-8') as infile:
        with open(f'{path}/clean/{title}.csv','w',encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['text'])
            writer.writeheader()
            for line in infile.readlines():               
                if line.strip() and re.search('[a-zA-Z]', line):
                    line = re.sub(u'“', "\"", line)
                    line = re.sub(u'’', "\'", line)
                    line = re.sub(u'‘', "\'", line)
                    line = re.sub(u'”', "\"", line)
                    writer.writerow({'text': line})                      


datasets = load_dataset('csv', 
                       data_files=[f'{path}/clean/{title}.csv' for title in titles])




tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def tokenize_function(examples):
    return tokenizer(examples["text"])

column_names = datasets["train"].column_names
tokenized_datasets = datasets.map(tokenize_function, 
                                  batched=True,
                                  remove_columns=column_names
                                  )

block_size = 1024

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size]
            for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts,
                                     batched=True,
                                     batch_size=100,   
)


config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             config=config)

train_dataset = lm_datasets["train"]

# Initialize Trainer
train_args = TrainingArguments(output_dir=model_path)
# Need: pip install git-lfs

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)


# Training
train_result = trainer.train()

# Save results
trainer.save_model(output_dir=model_path) 

metrics = train_result.metrics

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
