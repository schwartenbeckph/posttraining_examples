# Code to illustrate the sft process.

# We start with imports and managing warnings:

# pip install transformers
# warning control
import warnings

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = {
    "checkpoint": {"HuggingFaceTB/SmolLM2-135M"},
    "device": {"cpu": 8},
}

warnings.filterwarnings('ignore')

def load_config(config_path: str) -> DictConfig:
    """
    Load a config YAML file using omegaconf.
    :param: config_path (str): The path to the config YAML file.
    :return: omegaconf.DictConfig: The loaded config as a DictConfig object.
    """
    with open(config_path, "r") as file:
        config = OmegaConf.load(file)

    config = OmegaConf.merge(OmegaConf.create(DEFAULT_CONFIG), config)
    return config

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