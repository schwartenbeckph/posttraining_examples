# Code to illustrate the sft process.

# We start with imports and managing warnings:

# pip install transformers
# warning control
import warnings

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from sft_util import generate_responses, test_model_with_questions, load_model_and_tokenizer, display_dataset, load_config
from sft_static import QUESTIONS

warnings.filterwarnings('ignore')

# SFTTrainer config 
sft_config = SFTConfig(
    learning_rate=8e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
)

if __name__ == "__main__":
    print("I can show you how to SFT:")
    configs = load_config(config_path="src/sft_config.yaml")

    print("Define Checkpoint and device:")
    checkpoint = configs.checkpoint
    device = configs.device # for GPU usage or "cpu" for CPU usage

    print("Load model and tokeniser:")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    print("Test model before SFT:")
    test_model_with_questions(model, tokenizer, QUESTIONS, 
                            title="Base Model (After SFT) Output")

    print("Load dataset:")
    train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
    # decrease size of training data for runtime issues
    # train_dataset=train_dataset.select(range(100))
    # train_dataset=train_dataset.select(range(40))
    train_dataset=train_dataset.select(range(10))
    display_dataset(train_dataset)

    print("Perform SFT:")
    sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset, 
    processing_class=tokenizer,
    )
    sft_trainer.train()

    print("Test model after SFT:")
    sft_trainer.model.to(device)
    test_model_with_questions(sft_trainer.model, tokenizer, QUESTIONS, 
                            title="Base Model (After SFT) Output")
