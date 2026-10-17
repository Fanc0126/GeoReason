"""Prompt formats for reasoning and referring segmentation."""

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask?",
    "What is {class_name} in this image? Please output segmentation mask?",
    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
]
ANSWERS = ["It is [SEG].", "Sure, [SEG].", "Sure, it is [SEG].",
           "Sure, the segmentation result is [SEG].", "[SEG]."]
SEG_INSTRUCTIONS = ["Please segment the mentioned object.", "Show me the segmentation mask of it.",
                    "Locate it with a mask.", "Identify and segment the specific area.",
                    "Provide the segmentation result for this."]


def segmentation_prompt(question, task, ap=True, bbox_cot=True, evaluation=False):
    answer = "answer with [SEG] here" if ap else "answer here"
    if task == "reasoning":
        reasoning = "your detailed reasoning" if evaluation else "your reasoning"
        content = reasoning + (" and the bbox of the target in JSON format" if bbox_cot else "")
        example = "reasoning and bbox here" if bbox_cot else "reasoning here"
    elif bbox_cot:
        content, example = "the bbox of the target in JSON format", "bbox here"
        if evaluation and ap:
            answer = "answers with [SEG] here"
    else:
        return question + (" [SEG]" if ap else "")
    instruction = (f"When responding, please output {content} in <think> </think> tags, "
                   "and then provide your answer in <answer> </answer> tags, "
                   f"i.e., <think> {example} </think> <answer> {answer} </answer>.")
    separator = "\n\n" if evaluation and task == "reasoning" else "\n" if evaluation else " "
    return question + separator + instruction


def segmentation_answer(reasoning, answer, bbox_text, task, bbox_cot=True):
    if task == "referring" and not bbox_cot:
        return answer
    think = reasoning.strip()
    if think and not think.endswith((".", "!", "?")):
        think += "."
    if bbox_cot:
        think = (think + " " if think else "") + "The bbox of the target: " + bbox_text
    return f"<think> {think} </think>\n<answer> {answer} </answer>"
