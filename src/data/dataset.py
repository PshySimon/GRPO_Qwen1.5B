"""数据集处理"""

import random

from datasets import load_dataset

from .utils import SYSTEM_PROMPT, build_prompt, extract_answer_from_dataset


def prepare_dataset(split="train"):
    """
    Load and prepare the GSM8K dataset for training with string prompts.

    Args:
        split (str): The dataset split to load ("train" or "test"). Defaults to "train".

    Returns:
        list: A list of formatted examples, each containing a prompt string and answer.

    Explanation:
        1. Loads the GSM8K dataset from the Hugging Face datasets hub.
        2. For each example in the dataset:
           - Creates a list of messages with system prompt and the question.
           - Converts this list into a single string prompt using build_prompt().
           - Extracts the answer from the dataset example.
           - Creates a formatted example dictionary with prompt and answer.
        3. Returns the list of formatted examples ready for model training or evaluation.
    """
    data = load_dataset("openai/gsm8k", "main", cache_dir="./datasets")[split]
    formatted_data = []
    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)
    return formatted_data


def split_dataset(all_data, eval_size=50):
    """
    分割数据集为训练集和评估集

    Args:
        all_data (list): 完整数据集
        eval_size (int): 评估集大小

    Returns:
        tuple: (train_data, eval_data)
    """
    random.shuffle(all_data)
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    return train_data, eval_data
