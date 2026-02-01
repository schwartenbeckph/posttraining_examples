# Code to illustrate the grpo process.

# We start with imports and managing warnings:

# pip install transformers
# warning control
import warnings

import pandas as pd
import torch
import tqdm
from datasets import load_dataset, Dataset
import transformers
from trl import GRPOTrainer, GRPOConfig
from grpo_util import generate_responses, test_model_with_questions, load_model_and_tokenizer, load_config, reward_func, post_processing
from grpo_static import QUESTIONS, SYSTEM_PROMPT
import re

warnings.filterwarnings('ignore')

# GRPOTrainer config 
grpo_config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4, # Can set as high as 64 or 128
    num_train_epochs=1,
    learning_rate=5e-6,
    logging_steps=2,
    no_cuda=True    # keeps the whole run on CPU, incl. MPS
)

if __name__ == "__main__":
    print("I can show you how to GRPO:")
    configs = load_config(config_path="src/grpo_config.yaml")

    print("Define Checkpoint and device:")
    checkpoint = configs.checkpoint
    device = configs.device # for GPU usage or "cpu" for CPU usage

    print("Load model and tokeniser:")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    print("Load evaluation dataset")
    data_num = 5
    eval_dataset = load_dataset("openai/gsm8k", "main")["test"].select(range(data_num))
    sample_df = eval_dataset.to_pandas()
    eval_dataset = eval_dataset.map(post_processing).remove_columns(["question", "answer"])

    print("Test model before GRPO:")
    all_preds = []
    all_labels = []

    for example in tqdm(eval_dataset):
        input_prompt = example["prompt"]
        ground_truth = example["ground_truth"]
        # Run the model to generate an answer
        with torch.no_grad():
            response = generate_responses(model, tokenizer, 
                                        full_message = input_prompt) 
        all_preds.append([{"role": "assistant", "content": response}])
        all_labels.append(ground_truth)
        print(response)
        print("Ground truth: ", ground_truth)

    # 3. Evaluate using reward_func
    rewards = reward_func(all_preds, all_labels)

    # 4. Report accuracy
    accuracy = sum(rewards) / len(rewards)
    print(f"Evaluation Accuracy: {accuracy:.2%}")
    del model, tokenizer

    print("Load dataset:")
    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"]
    # Apply to dataset
    train_dataset = train_dataset.map(post_processing)
    train_dataset = train_dataset.remove_columns(["question", "answer"])
    train_dataset = train_dataset.select(range(10))

    print("Perform GRPO:")
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=train_dataset
    )
    grpo_trainer.train()

    print("Test model after GRPO:")
    grpo_trainer.model.to(device)
    # Store predictions and ground truths
    all_preds = []
    all_labels = []

    for example in tqdm(eval_dataset):
        input_prompt = example["prompt"]
        ground_truth = example["ground_truth"]
        # Run the model to generate an answer
        with torch.no_grad():
            response = generate_responses(grpo_trainer, tokenizer,
                                        full_message = input_prompt) 
        all_preds.append([{"role": "assistant", "content": response}])
        all_labels.append(ground_truth)
        print(response)
        print("Ground truth: ", ground_truth)

    # 3. Evaluate using reward_func
    rewards = reward_func(all_preds, all_labels)

    # 4. Report accuracy
    accuracy = sum(rewards) / len(rewards)
    print(f"Evaluation Accuracy: {accuracy:.2%}")
