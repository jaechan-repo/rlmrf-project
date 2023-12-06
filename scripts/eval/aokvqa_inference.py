import sys

from spacy.lang.en import English
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import os
from rampa.llava.model.builder import load_pretrained_model
from rampa.llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, get_model_name_from_path
from rampa.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from rampa.llava.conversation import conv_templates, SeparatorStyle
from rampa.passage_retriever import ContrieverRetriever
from rampa.perplexity import LLaVAPerplexity
from rampa.utils import LLaVAProcessor
from rampa.mytyping import DataQA, Response, Doc, Question, LLaVAFinetuneData
import json
import math
from typing import Dict, List, Literal
from tqdm import tqdm
from PIL import Image
import torch
import shortuuid
import argparse
from config import BASE_DIR
import re

nlp = English()
nlp.add_pipe('sentencizer')


def index_to_letter(choices):
    """ Convert zero-based indices to letters. """
    return {index: f"({chr(65 + index)})" for index in range(len(choices))}


def letter_to_index(choices):
    """ Convert letters to zero-based indices. """
    return {f"({chr(65 + index)})": index for index in range(len(choices))}


def format_choices(choices):
    """ Format the choices in the specified format. """
    index_letter_map = index_to_letter(choices)
    formatted_choices = " ".join(f"{index_letter_map[index]} {choice}" for index, choice in enumerate(choices))
    return formatted_choices


def split_in_sentences(text: str) -> List[str]:
    """ Split the text into sentences. """
    return [str(sent).strip() for sent in nlp(text).sents]


def find_correct_answer(response):
    """
    Identify the sentence containing "correct answer" and parse the letter.
    """
    # match = re.search(r"([A-D])(\)|.|\b|\n)", response)
    match = re.search(r"([A-D])\)", response)
    if match:
        return ord(match.group(1).upper()) - ord('A')
    return -1


def get_prompt_from_question(question: str, 
                             choices: List[str],
                             mm_use_im_start_end: bool, 
                             conv_mode: str):
    if mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + f"Question: {question}"
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + f"Question: {question}"
    # question += " Answer and explain."
    question += "\nOptions: " + format_choices(choices)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], "Answer: (")
    return conv.get_prompt().replace("</s>", "")


def rollout_model(student_path: str,
                  student_base: str,
                  dataset: str | List[DataQA],
                  image_folder: str,
                  max_new_tokens: int = 128,
                  conv_mode: str = 'llava_v1',
                  batch_size: int = 12) -> Dict:
    
    # Load model
    model_name = get_model_name_from_path(os.path.abspath(student_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
            student_path, student_base, model_name, load_4bit=True, device_map="auto")
    processor = LLaVAProcessor(tokenizer, 
                               image_processor, 
                               model.config.mm_use_im_start_end,
                               conv_mode)
    
    # Load dataset
    if isinstance(dataset, str):
        with open(dataset, 'r') as f:
            dataset: List[DataQA] = json.load(f)

    mc_submission = {}

    ### BEGIN: Batch processing (VQA-only) ###
    dataset_batch: List[List[DataQA]] = \
            [dataset[i:i+batch_size] for i in range(0,len(dataset),batch_size)]
    for batch in tqdm(dataset_batch):
        questions: List[Question] = [line['question'] for line in batch]
        image_files: List[str] = [
            os.path.join(image_folder, question['image_file']) \
            for question in questions]
        
        #  Get input_ids, image_tensor
        prompts = [get_prompt_from_question(question['text'], question['choices'], 
                                            model.config.mm_use_im_start_end, conv_mode) for question in questions]
        images = [LLaVAProcessor.load_image(image_file) for image_file in image_files]
        batch_input_ids = [
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts
        ]
        max_len = max([len(seq) for seq in batch_input_ids])
        padded_input_ids = [LLaVAProcessor.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)
        batch_image_tensor = image_processor(images, return_tensors="pt")["pixel_values"]
    
        # Establish stopping criteria
        conv = conv_templates[conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, batch_input_ids)]
            if conv.version == "v0"
            else None
        )

        with torch.inference_mode():
            batch_output_ids = model.generate(
                batch_input_ids.cuda(),
                images=batch_image_tensor.half().cuda(),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria
                )

        input_token_len = batch_input_ids.shape[1]
        batch_outputs = tokenizer.batch_decode(
            batch_output_ids[:, input_token_len:], skip_special_tokens=True)
        
        for i, line in enumerate(batch):
            output = batch_outputs[i:len(batch_outputs):len(batch)][0].strip()
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            parsed_index = find_correct_answer(output)
            print(output)
            print(f"Parsed number: {parsed_index}")
            print()
            mc_submission[line['question']['qid']] = {
                'multiple_choice': line['question']['choices'][parsed_index],
                'direct_answer': ""
            }

    del tokenizer, model, image_processor
    return mc_submission

# For evaluation
def main(args):
    mc_submission = rollout_model(
                student_path=args.model_path,
                student_base=args.model_base,
                dataset=args.dataset_file,
                image_folder=args.image_folder,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(mc_submission, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### SUBJECT TO CHANGE ###
    parser.add_argument("--dataset-file", type=str, 
                        default=os.path.join(BASE_DIR, "data_in/iter0/aokvqa/test.json"))
    parser.add_argument("--output-file", type=str, 
                        default=os.path.join(BASE_DIR, "data_out/aokvqa/reward_image/predictions_test.json"))
    parser.add_argument("--model-path", type=str, 
                        default=os.path.join(BASE_DIR, "ckpt/iter1/reward_image/llava-v1.5-7b-lora"))
    parser.add_argument("--model-base", type=str, 
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    # parser.add_argument("--model-path", type=str, 
    #                     default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    # parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--task", type=str, default='aokvqa')
    parser.add_argument("--image-folder", type=str, 
                        default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--conv-mode", type=str, default='llava_v1')
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    main(args)