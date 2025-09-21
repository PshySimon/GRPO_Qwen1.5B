"""评估脚本"""

import torch
from src.data.dataset import prepare_dataset
from src.data.utils import (
    extract_answer_from_model_output,
    extract_last_number,
    extract_single_number,
)
from src.utils.device import get_device
from src.utils.logging import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints detailed results.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.

    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).

    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    logger = setup_logger()

    logger.info("\n" + "=" * 50)
    logger.info(f"EVALUATION ON {total} EXAMPLES")
    logger.info("=" * 50)

    for example in eval_examples:
        # Get the prompt and expected answer
        full_prompt = example["prompt"]
        expected = example["answer"]

        # Tokenize and generate response
        inputs = tokenizer.encode(
            full_prompt, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Extract answer and check correctness
            predicted = extract_answer_from_model_output(response)

            # Try different matching methods
            if predicted == expected:  # Exact match
                is_correct = True
            else:
                # Try single number matching
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number matching
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (
                        pred_num is not None
                        and exp_num is not None
                        and pred_num == exp_num
                    )

            # Update counter for correct answers
            if is_correct:
                correct += 1

            # Print evaluation details
            logger.info("\nPrompt:")
            logger.info(full_prompt)
            logger.info("\nExpected Answer:")
            logger.info(expected)
            logger.info("\nExtracted Answer:")
            logger.info(predicted)
            logger.info("\nFull Generated Response:")
            logger.info(response)
            if is_correct:
                logger.info("\nCorrect:✓")
            else:
                logger.info("\nCorrect:✗")
            logger.info("-" * 50)

        except Exception as e:
            logger.info("\nFailed to parse model output for prompt:")
            logger.info(full_prompt)
            logger.info("Error:", e)
            logger.info("-" * 50)

    # Calculate and print final accuracy
    accuracy = (correct / total) * 100
    logger.info(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info("=" * 50)

    # Return model to training mode
    model.train()
    return accuracy


def main():
    """主评估函数"""
    # 设置设备
    device = get_device()

    # 加载模型和tokenizer
    model_name = "/Users/caixiaomeng/Projects/Python/GRPO_Qwen2.5-1.5B/grpo_qwen_1.5b/models/qwen2.5-1.5b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # 设置tokenizer配置
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 准备评估数据
    all_data = prepare_dataset("train")
    eval_data = all_data[:50]  # 使用前50个样本进行评估

    # 评估模型
    accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Model Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
