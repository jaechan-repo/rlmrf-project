import sys
sys.path.append("..")

import os
from rampa.llava.model.builder import load_pretrained_model
from rampa.llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from rampa.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from rampa.llava.conversation import conv_templates, SeparatorStyle
from rampa.passage_retriever import ContrieverRetriever
from rampa.perplexity import LLaVAPerplexity
import json
import math
from rampa.mytyping import DataQA, Response, Doc, Question, LLaVAFinetuneData
from typing import List, Dict, Literal
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from numpy.typing import NDArray
import shortuuid
import argparse
import functools
from rampa.config import BASE_DIR
from spacy.lang.en import English

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def pad_image_token(q: str, mm_use_im_start_end: bool) -> str:
    if mm_use_im_start_end:
        q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
    else:
        q = DEFAULT_IMAGE_TOKEN + '\n' + q
    return q

def rollout_model(student_path: str,
                  student_base: str,
                  dataset: str | List[DataQA],
                  output_file: str,
                  image_folder: str,
                  num_return_sequences: int,
                  conv_mode: str = 'llava-v1',
                  temperature: float = 0.9,
                  top_p: float = 1.0,
                  num_beams: int = 1,
                  num_chunks: int = 1,
                  chunk_idx: int = 0):
    
    # Load model
    model_name = get_model_name_from_path(os.path.abspath(student_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
            student_path, student_base, model_name, load_4bit=True)
    
    # Load dataset
    if isinstance(dataset, str):
        dataset: List[DataQA] = json.load(dataset)
    dataset = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_dataset: List[DataQA] = []

    for line in tqdm(dataset):
        question: Question = line['question']
        image_file = question['image_file'] # WIP: the query may not contain an image
        qs = question['text']

        ### BEGIN: copied from run_llava.py ###

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

    json.dump(new_dataset, output_file)

    del tokenizer, model, image_processor
    return new_dataset

def retrieve_docs(passages: str,
                  passages_embeddings: str,
                  retriever_path: str,
                  dataset: str | List[DataQA],
                  output_file: str,
                  num_chunks: int = 1,
                  chunk_idx: int = 0,
                  n_docs: int = 3):

    retriever = ContrieverRetriever(passages=passages)
    retriever.prepare_model(model_path=retriever_path, 
                            passages_embeddings=passages_embeddings)
    # Load dataset
    if isinstance(dataset, str):
        dataset = json.load(dataset)
    dataset: List[DataQA] = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    queries: List[str] = [line['response']['text'] + " " + line['question']['text'] \
                    for line in dataset]
    doc_ids_list = retriever.retrieve_doc_ids(queries, n_docs=n_docs)

    new_dataset = []
    for i, line in enumerate(dataset):
        doc_ids, doc_scores = doc_ids_list[i]
        docs: List[Doc] = [{
            'doc_id': doc_ids[j], 
            'doc_score': doc_scores[j]
        } for j in range(len(doc_ids))]
        line['docs']: List[Doc] = [doc | retriever.id_to_text(doc) for doc in docs]
        new_dataset.append(line)
    
    json.dump(new_dataset, output_file)
    del retriever
    return new_dataset
    

def compute_perplexity(teacher_path: str,
                       dataset: str | List[DataQA],
                       output_file: str,
                       image_folder: str,
                       num_chunks: int = 1,
                       chunk_idx: int = 0,
                    #    doc_condition_mode: Literal['concat', 'marginal'] = 'concat',
                       conv_mode: str = 'llava-v1',
                       lambda_const: float = 0.4
                       )-> List[DataQA]:
    
    # Load dataset
    if isinstance(dataset, str):
        dataset = json.load(dataset)
    dataset: List[DataQA] = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Perplexity grading
    teacher = LLaVAPerplexity(model_path=teacher_path, load_4bit=True)

    # Sentencizer
    nlp = English()
    nlp.add_pipe('sentencizer')
    def split_in_sentences(text: str) -> List[str]:
        return [str(sent).strip() for sent in nlp(text).sents]

    new_dataset: List[DataQA] = []

    for line in dataset:
        question, docs, responses = line['question'], line['docs'], line['responses']
        if 'image_file' in line['question'] and line['question']['image_file'] is not None:
            image = Image.open(os.path.join(image_folder, line['question']['image_file']))
        else:
            image = None

        ### 1. Compute sentence-level ppl for each response ###
        ppls_sent = []
        for response in responses:
            sentences = split_in_sentences(response['text'])
            prompts = []
            for sentence in sentences:
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], question['text'])
                conv.append_message(conv.roles[1], sentence)
                prompts.append(conv.get_prompt())
            if image is not None:
                images = [image.copy() for _ in range(len(sentences))]
                ppls = teacher.compute_visual(prompts, images)['perplexities']
            else:
                ppls = teacher.compute(prompts)['perplexities']
            ppl = functools.reduce(lambda x, y: x*y, ppls) # independence assumption
            ppls_sent.append(ppl)

        ### 2. Compute retrieval-augmented ppl for each response ###
        doc_texts: List[str] = [doc['text'] for doc in docs]
        doc_concat: str = '\n\n'.join(doc_texts)
        question_doc: str = pad_image_token(doc_concat + "\n\nQuestion: " + question['text'])
        R = len(responses)
        prompts = []
        for response in responses:
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question_doc)
            conv.append_message(conv.roles[1], response['text'])
            prompts.append(conv.get_prompt())
        if image is not None:
            images = [image.copy() for _ in range(R)]
            ppls_rag = teacher.compute_visual(prompts, images)['perplexities']
        else:
            ppls_rag = teacher.compute(prompts)['perplexities']

        ### 3. Merge ###
        ppls = [lambda_const * p1 + p2 for p1,p2 in zip(ppls_sent, ppls_rag)]
        rewards = [-ppl for ppl in ppls]   # invert
        line['responses'] = [line['responses'][i] | {'reward':rewards[i]} for i in range(R)]
        line['top_response'] = max(line['responses'], key=lambda r: r['reward'])
        new_dataset.append(line)
    
    json.dump(new_dataset, output_file)
    del teacher
    return new_dataset

def make_sft_dataset(dataset: str | List[DataQA],
                     image_folder: str,
                     output_file: str) -> List[LLaVAFinetuneData]:
    """Creates an SFT dataset that can be directly fed to LLaVA for fine-tuning.
    Follows the format of: 
    https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
    """
    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if isinstance(dataset, str):
        dataset: List[DataQA] = json.load(dataset)
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
    json.dump(sft_dataset, output_file)
    return sft_dataset

def main(args) -> List[DataQA]:
    dataset: str = os.path.join(args.dataset_folder, args.dataset_name, f"{args.split}.json")
    if not os.path.isfile(dataset):
        raise FileNotFoundError
    dataset: List[DataQA] = rollout_model(student_path=args.rollout_student_path,
                                          student_base=args.rollout_student_base,
                                          dataset=dataset,
                                          output_file=os.path.join(args.output_folder, 
                                                                   args.dataset_name, 
                                                                   f"rollout/{args.split}.json"),
                                          image_folder=args.image_folder,
                                          num_return_sequences=args.rollout_num_return_sequences,
                                          conv_mode=args.conv_mode,
                                          num_chunks=args.num_chunks,
                                          chunk_idx=args.chunk_idx)
    dataset: List[DataQA] = retrieve_docs(passages=args.retrieval_passages,
                                          passages_embeddings=args.retrieval_passages_embeddings,
                                          retriever_path=args.retrieval_retriever_path,
                                          dataset=dataset,
                                          output_file=os.path.join(args.output_folder, 
                                                                   args.dataset_name, 
                                                                   f"retrieval/{args.split}.json"),
                                          n_docs=args.retrieval_n_docs)
    dataset: List[DataQA] = compute_perplexity(teacher_path=args.reward_teacher_path,
                                               dataset=dataset,
                                               output_file=os.path.join(args.output_folder, 
                                                                        args.dataset_name, 
                                                                        f"reward/{args.split}.json"),
                                               image_folder=args.image_folder,
                                               conv_mode=args.conv_mode,
                                               lambda_const=args.reward_lambda_const)
    dataset: List[DataQA] = make_sft_dataset(dataset=dataset,
                                             image_folder=args.image_folder,
                                             output_file=os.path.join(args.output_folder, 
                                                                      args.dataset_name, 
                                                                      f"rs/{args.split}.json"))
    return dataset
                                    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dataset-folder", type=str, default=os.path.join(BASE_DIR, "data_prep/iter0")) # SUBJECT TO CHANGE
    parser.add_argument("--image-folder", type=str, default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--output-folder", type=str, default=os.path.join(BASE_DIR, "data_prep/iter1")) # SUBJECT TO CHANGE
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default='llava-v1')

    ### Rollout ###
    parser.add_argument("--rollout-student-path", type=str, 
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    parser.add_argument("--rollout-student-base", type=str, 
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    parser.add_argument("--rollout-num-return-sequences", type=int, default=5)

    ### Retrieval ###
    parser.add_argument("--retrieval-passages", type=str, 
                        default=os.path.join(BASE_DIR, "data/contriever_msmarco/psgs_w100.tsv"))
    parser.add_argument("--retrieval-passages-embeddings", type=str,
                        default=os.path.join(BASE_DIR, "data/contriever_msmarco/wikipedia_embeddings/passages_*"))
    parser.add_argument("--retrieval-retriever-path", type=str,
                        default=os.path.join(BASE_DIR, "models/contriever"))
    parser.add_argument("--retrieval-n-docs", type=int, default=3)
    
    ### Reward ###
    parser.add_argument("--reward-teacher-path", type=str,
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    parser.add_argument("--reward-lambda-const", type=float, default=0.4)
    args = parser.parse_args()

    if args.dataset_name is None:
        for dataset_name in ['aokvqa', 's3vqa', 'scienceqa', 'okvqa']:
            args.dataset_name = dataset_name
            if args.split is None:
                for split in ['train', 'val']:
                    args.split = split
                    main(args)
            else:
                main(args)      
    else:
        if args.split is None: 
            for split in ['train', 'val']:
                args.split = split
                main(args)
        else:
            main(args)
