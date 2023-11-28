"""
Warning:
Everything here is 4-bit!
"""

import os
from rlmrf.llava.model.builder import load_pretrained_model
from rlmrf.llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from rlmrf.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from rlmrf.llava.conversation import conv_templates, SeparatorStyle
from rlmrf.passage_retriever import ContrieverRetriever
from rlmrf.perplexity import LLaVAPerplexity
import json
import math
from rlmrf.typing import DataQA, Response, Doc, Question, LLaVAFinetuneData
from typing import List, Dict, Literal
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from numpy.typing import NDArray
import shortuuid
import argparse

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def dataset_loader(dataset_file: str) -> List[DataQA]:
    return [json.loads(line) for line in open(dataset_file, 'r')]

def pad_image_token(q: str, mm_use_im_start_end: bool) -> str:
    if mm_use_im_start_end:
        q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
    else:
        q = DEFAULT_IMAGE_TOKEN + '\n' + q
    return q

def rollout_model(model_path: str,
                  model_base: str,
                  dataset: str | List[DataQA],
                  output_file: str,
                  image_folder: str,
                  num_return_sequences: int,
                  conv_mode: str = 'llava-v1',
                  temperature: float = 0.8,
                  top_p: float = 1.0,
                  num_beams: int = 1,
                  num_chunks: int = 1,
                  chunk_idx: int = 0):
    
    # Load model
    model_name = get_model_name_from_path(os.path.abspath(model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, load_4bit=True)
    
    # Load dataset
    if isinstance(dataset, str):
        dataset: List[DataQA] = dataset_loader(dataset)
    dataset = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out = open(output_file, 'w')

    new_dataset: List[DataQA] = []

    for line in tqdm(dataset):
        question: Dict = line['question']
        image_file = question['image_file'] # WIP: the query may not contain an image
        
        qs = question['text']

        ### BEGIN: copied from run_llava.py

        # Build prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        ### END ###

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        responses: List[Response] = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            responses.append({
                'text': output,
                'rid': shortuuid.uuid()
            })
        line['responses'] = responses
        new_dataset.append(line)
        out.write(line)

    return new_dataset

def compute_perplexity(teacher_path: str,
                       dataset: str | List[DataQA],
                       output_file: str,
                       image_folder: str,
                       num_chunks: int = 1,
                       chunk_idx: int = 0,
                       passages_file: str = "data/psgs_w100.tsv",
                       passages_embeddings: str = "data/contriever_msmarco/wikipedia_embeddings/passages_*",
                       doc_condition_mode: Literal['concat', 'marginal'] = 'concat',
                       conv_mode: str = 'llava-v1',
                       )-> List[DataQA]:
    
    # Load dataset
    if isinstance(dataset, str):
        dataset: List[DataQA] = dataset_loader(dataset)
    dataset = get_chunk(dataset, num_chunks, chunk_idx)

    def softmax(scores: List[float], gamma: float = 0.5) -> NDArray:
        scores = np.array(scores) * gamma
        scores_exp = np.exp(scores)
        return scores_exp / np.sum(scores_exp)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out = open(output_file, 'w')

    # Perplexity grading
    teacher = LLaVAPerplexity(model_path=teacher_path, load_4bit=True)
    retriever = ContrieverRetriever(passages_file, passages_embeddings)
    
    new_dataset: List[DataQA] = []
    for line in dataset:

        R, D = len(line['responses']), len(docs)
        doc_texts = [retriever.id_to_text(doc['doc_id']) for doc in line['docs']]
        docs = [line['docs'][i] | doc_texts[i] for i in range(D)]
        image = Image.open(os.path.join(image_folder, line['question']['image_file']))

        if doc_condition_mode == 'marginal':
            prompts = []    # D * R number of prompts
            for doc in docs:
                question_text = line['question']['text']
                question_text = pad_image_token(doc + "\n\nQuestion: " + question_text)
                for response in line['responses']:
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question_text)
                    conv.append_message(conv.roles[1], response['text'])
                    prompts.append(conv.get_prompt())

            images = [image.copy() for _ in range(D*R)]
            ppls = np.array(teacher.compute(prompts, images, batch_size=R)['perplexities']).reshape(D, R)
            p_doc = softmax(line['doc_scores'])
            rewards = (np.asarray(-ppls@p_doc)).tolist()
        
        elif doc_condition_mode == 'concat':
            doc_concat = '\n\n'.join(docs)
            question_text = line['question']['text']
            question_text = pad_image_token(doc_concat + "\n\nQuestion: " + question_text)

            prompts = []
            for response in line['responses']:
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], question_text)
                conv.append_message(conv.roles[1], response['text'])
                prompts.append(conv.get_prompt())
                
            images = [image.copy() for _ in range(R)]    # same image because same query
            rewards = teacher.compute(prompts, images, batch_size=R)['perplexities']
            rewards = [-reward for reward in rewards]   # invert

        else:
            raise NotImplementedError
    
        line['responses'] = [line['responses'][i] | {'reward':rewards[i]} for i in range(R)]
        line['top_response'] = max(line['responses'], key=lambda r: r['reward'])
        new_dataset.append(line)
        out.write(line)
    return new_dataset

# follows the format of: https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
def make_sft_dataset(dataset: str | List[DataQA],
                     image_folder: str) -> List[LLaVAFinetuneData]:
    if isinstance(dataset, str):
        dataset: List[DataQA] = dataset_loader(dataset)
    sft_dataset = []
    for line in dataset:
        image_file = os.path.join(image_folder, line['question']['image_file'])
        finetune_data: LLaVAFinetuneData = {
            'id': line['top_response']['rid'],
            'image': image_file,
            'conversations': [
                {
                    'from': 'human',
                    'value': line['question']['text']
                },
                {
                    'from': 'gpt',
                    'value': line['top_response']['text']
                }
            ]
        }
        sft_dataset.append(finetune_data)
    return sft_dataset

if __name__ == '__main__':
    raise NotImplementedError

    parser = argparse.ArgumentParser()

    # Rollout
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()


