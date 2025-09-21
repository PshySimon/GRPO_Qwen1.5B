"""数据处理工具函数"""

import re

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.

    Args:
        text (str): The model-generated text containing XML-style <answer> tags.

    Returns:
        str or None: The content inside the <answer> tags, or None if no valid answer is found.

    Explanation:
        1. Splits the text on the <answer> tag to isolate content after the tag.
        2. Checks if at least one <answer> tag exists in the text.
        3. For the last <answer> segment:
           - Verifies it contains a closing </answer> tag.
           - Extracts only the content between the tags.
        4. Returns None if the answer is empty (just "...") or if tags are missing.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the GSM8K dataset examples.

    Args:
        text (str): The dataset example text containing a question and answer.

    Returns:
        str or None: The extracted answer part after the '####' delimiter, or None if not found.

    Explanation:
        1. Checks if the text contains the '####' delimiter that separates question from answer.
        2. If found, splits the text at this delimiter and returns the second part (the answer).
        3. The answer is stripped of leading/trailing whitespace.
        4. Returns None if no delimiter is present.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_last_number(text):
    """
    Extracts the last number appearing in the text.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The last number in the text, or None if no number is found.

    Explanation:
        1. Removes dollar signs and percent symbols from the text.
        2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
        3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
        4. Returns the found number as a float, or None if no match is found.
    """
    text = text.replace("$", "").replace("%", "")
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    """
    Extracts a single number from text if exactly one number is present.

    Args:
        text (str): The text to extract a number from.

    Returns:
        float or None: The single number in the text, or None if zero or multiple numbers are found.

    Explanation:
        1. Uses regex to find all numbers in the text (including negative numbers and decimals).
        2. If exactly one number is found, returns it as a float.
        3. If zero or multiple numbers are found, returns None.
    """
    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.

    Args:
        messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

    Returns:
        str: A concatenated string of all message contents.

    Explanation:
        1. Takes a list of message dictionaries in the typical chat format.
        2. Extracts the 'content' field from each message and strips whitespace.
        3. Joins all content strings with newlines to create a single prompt.
        4. This preserves the training format while converting from structured messages to a string.
    """
    return "\n".join([msg["content"].strip() for msg in messages])
