# Code to illustrate the dpo process.

# We start with imports and managing warnings:

# pip install transformers
# warning control
import warnings

import pandas as pd
import torch
import tqdm
from datasets import load_dataset
import transformers
from trl import DPOTrainer, DPOConfig
from dpo_util import test_model_with_questions, load_model_and_tokenizer, load_config
from dpo_static import QUESTIONS


warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

# DPOTrainer config 
dpo_config = DPOConfig(
    beta=0.2, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=2,
)

if __name__ == "__main__":
    print("I can show you how to DPO:")
    configs = load_config(config_path="src/dpo_config.yaml")

    print("Define Checkpoint and device:")
    checkpoint = configs.checkpoint
    device = configs.device # for GPU usage or "cpu" for CPU usage

    print("Load model and tokeniser:")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    print("Test model before DPO:")
    test_model_with_questions(model, tokenizer, QUESTIONS, 
                            title="Base Model (before DPO) Output")

    print("Load dataset:")
    dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")
    # dpo_ds = dpo_ds.select(range(100))
    dpo_ds = dpo_ds.select(range(10))

    print("Perform DPO:")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=dpo_ds
    )

    dpo_trainer.train()

    print("Test model after DPO:")
    dpo_trainer.model.to(device)
    test_model_with_questions(dpo_trainer.model, tokenizer, QUESTIONS,
                            title="Base Model (After DPO) Output")
