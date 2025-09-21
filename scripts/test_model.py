"""测试微调后的模型"""

import torch
from src.data.utils import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output
from src.utils.device import get_device
from src.utils.logging import setup_logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_fine_tuned_model():
    """
    Main function to load the fine-tuned model and test it on example math problems.

    Explanation:
        1. Determines the device (GPU if available, otherwise CPU).
        2. Loads the fine-tuned model and tokenizer from the saved path.
        3. Tests the model on predefined math problems.
        4. Formats the prompt using the same SYSTEM_PROMPT and build_prompt function as training.
        5. Generates and displays responses for each test prompt.
    """
    # 确定设备
    device = get_device()
    logger = setup_logger()
    logger.info(f"Using device: {device}")

    # 加载保存的模型和tokenizer
    saved_model_path = "grpo_finetuned_model"

    # 加载模型
    loaded_model = AutoModelForCausalLM.from_pretrained(
        saved_model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    loaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

    # 定义测试提示
    prompts_to_test = [
        "How much is 1+1?",
        "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
        "Solve the equation 6x + 4 = 40",
    ]

    # 测试每个提示
    for prompt in prompts_to_test:
        # 准备提示，使用与训练期间相同的格式
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        test_prompt = build_prompt(test_messages)

        # 对提示进行tokenize并生成响应
        test_input_ids = loaded_tokenizer.encode(
            test_prompt, return_tensors="pt", padding=True, padding_side="left"
        ).to(device)

        # 使用与训练中类似的参数生成响应
        with torch.no_grad():
            test_output_ids = loaded_model.generate(
                test_input_ids,
                max_new_tokens=400,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=loaded_tokenizer.pad_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=False,
            )

        test_response = loaded_tokenizer.decode(
            test_output_ids[0], skip_special_tokens=True
        )

        # 打印测试提示和模型的响应
        logger.info("\nTest Prompt:")
        logger.info(test_prompt)
        logger.info("\nModel Response:")
        logger.info(test_response)

        # 提取并显示答案部分，便于评估
        try:
            extracted_answer = extract_answer_from_model_output(test_response)
            logger.info("\nExtracted Answer:")
            logger.info(extracted_answer)
            logger.info("-" * 50)
        except Exception as e:
            logger.info(f"\nFailed to extract answer: {e}")
            logger.info("-" * 50)


def main():
    """主函数"""
    test_fine_tuned_model()


if __name__ == "__main__":
    main()
