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

warnings.filterwarnings('ignore')

def generate_responses(model, tokenizer, user_message, system_message=None, max_new_tokens=100):
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    # outputs = model.generate(inputs)
    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids=outputs[0][input_len:]
    response=tokenizer.decode(generated_ids,skip_special_tokens=True).strip()
    return response

def test_model_with_questions(model, tokenizer, questions, 
                              system_message=None, title="Model Output"):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, 
                                      system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")

def load_model_and_tokenizer(checkpoint, device):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # if use_gpu:
    #     model.to("cuda")
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def display_dataset(dataset):
    # Visualize the dataset 
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages']
                        if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages']
                             if m['role'] == 'assistant')
        rows.append({
            'User Prompt': user_msg,
            'Assistant Response': assistant_msg
        })
    # Display as table
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)  # Avoid truncating long strings
    print(df)

# SFTTrainer config 
sft_config = SFTConfig(
    learning_rate=8e-5, # Learning rate for training. 
    num_train_epochs=1, #  Set the number of epochs to train the model.
    per_device_train_batch_size=1, # Batch size for each device (e.g., GPU) during training. 
    gradient_accumulation_steps=8, # Number of steps before performing a backward/update pass to accumulate gradients.
    gradient_checkpointing=False, # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
    logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
)

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What's the difference between thread and process?"
]

if __name__ == "__main__":
    print("I can show you how to SFT:")

    print("Define Checkpoint and device:")
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = "cpu" # for GPU usage or "cpu" for CPU usage

    print("Load model and tokeniser:")
    model, tokenizer = load_model_and_tokenizer(checkpoint, device)

    print("Test model before SFT:")
    test_model_with_questions(model, tokenizer, questions, 
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
    test_model_with_questions(sft_trainer.model, tokenizer, questions, 
                            title="Base Model (After SFT) Output")
