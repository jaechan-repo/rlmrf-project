import sys
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
from typing import List, Literal
from tqdm import tqdm
from PIL import Image
import torch
import shortuuid
import argparse
from config import BASE_DIR

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def rollout_model(student_path: str,
                  student_base: str,
                  dataset: str | List[DataQA],
                  output_file: str,
                  image_folder: str,
                  num_return_sequences: int,
                  max_new_tokens: int = 128,
                  conv_mode: str = 'llava_v1',
                  do_sample: bool= True,
                  temperature: float = 0.9,
                  top_p: float = 1.0,
                  num_chunks: int = 1,
                  chunk_idx: int = 0,
                  batch_size: int = 10):
    
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
            dataset = json.load(f)
    dataset: List[DataQA] = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_dataset: List[DataQA] = []

    ### BEGIN: Batch processing (VQA-only) ###
    dataset_batch: List[List[DataQA]] = \
            [dataset[i:i+batch_size] for i in range(0,len(dataset),batch_size)]
    for batch in tqdm(dataset_batch):
        questions: List[Question] = [line['question'] for line in batch]
        question_texts: List[str] = [question['text'] for question in questions]
        image_files: List[str] = [
            os.path.join(image_folder, question['image_file']) \
            for question in questions]
        batch_input_ids, batch_image_tensor = \
            processor.get_processed_tokens_batch(question_texts, image_files)
        conv = conv_templates[processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, processor.tokenizer, batch_input_ids)]
            if conv.version == "v0"
            else None
        )
        with torch.inference_mode():
            if do_sample:
                batch_output_ids = model.generate(
                    batch_input_ids.cuda(),
                    images=batch_image_tensor.half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria
                    )
            else:
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
            outputs = batch_outputs[i:len(batch_outputs):len(batch)]
            line['responses'] = []
            for output in outputs:
                output = output.strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                line['responses'].append({
                    'text': output,
                    'rid': shortuuid.uuid()
                })
            new_dataset.append(line)

    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(new_dataset, f)

    del tokenizer, model, image_processor
    return new_dataset

# For evaluation
def main(args):
    output: List[DataQA] = rollout_model(
                  student_path=args.model_path,
                  student_base=args.model_base,
                  dataset=args.dataset_file,
                  output_file=None,
                  image_folder=args.image_folder,
                  num_return_sequences=1,
                  do_sample=False,
                  conv_mode=args.conv_mode,
                  max_new_tokens=args.max_new_tokens,
                  batch_size=args.batch_size)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    f = open(args.output_file, 'w')
    if args.task == 'aokvqa':
        submission = []
        for line in output:
            submission.append({
                line['question']['qid']: {
                    'multiple_choice': "", # TODO Add support for multiple choice answering, too
                    'direct_answer': line['responses'][0]['text']
                }
            })
        json.dump(submission, f)
    else:
        json.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### SUBJECT TO CHANGE ###
    parser.add_argument("--dataset-file", type=str, 
                        default=os.path.join(BASE_DIR, "data_in/iter0/aokvqa/test.json"))
    parser.add_argument("--output-file", type=str, 
                        default=os.path.join(BASE_DIR, "data_out/aokvqa_submit_reward_text.json"))
    parser.add_argument("--model-path", type=str, 
                        default=os.path.join(BASE_DIR, "ckpt/iter1/reward_text/llava-v1.5-7b-lora"))
    parser.add_argument("--model-base", type=str, 
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))

    parser.add_argument("--task", type=str, default='aokvqa')
    parser.add_argument("--image-folder", type=str, 
                        default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--conv-mode", type=str, default='llava_v1')
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    main(args)