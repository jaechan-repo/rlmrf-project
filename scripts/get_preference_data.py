import sys
sys.path.append(".")
sys.path.append("..")

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
import functools
from config import BASE_DIR
from spacy.lang.en import English
from pathlib import Path
from batch_inference import rollout_model

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

def retrieve_docs(passages: str,
                  passages_embeddings: str,
                  retriever_path: str,
                  dataset: str | List[DataQA],
                  output_file: str,
                  num_chunks: int = 1,
                  chunk_idx: int = 0,
                  n_docs: int = 3
                  ):

    retriever = ContrieverRetriever(passages=passages)
    retriever.prepare_model(model_path=retriever_path, 
                            passages_embeddings=passages_embeddings)
    # Load dataset
    if isinstance(dataset, str):
        with open(dataset, 'r') as f:
            dataset = json.load(f)
    dataset: List[DataQA] = get_chunk(dataset, num_chunks, chunk_idx)

    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    ### NEW CODE START       ###
    ### TODO Is use of caption 'legal'? ###
    queries: List[str] = [
        f"{line['ref_cap']['text']}, {line['question']['text']}" for line in dataset
    ]
    ret_docs_list = retriever.retrieve_doc_ids(queries, n_docs=n_docs)
    docs_list: List[List[Doc]] = []
    for doc_ids, doc_scores in ret_docs_list:
        docs_list.append([{
            'doc_id': doc_ids[i],
            'doc_score': float(doc_scores[i])
        } for i in range(n_docs)])
    ### NEW CODE END ###

    new_dataset = []
    for i, line in enumerate(dataset):
        docs: List[Doc] = docs_list[i]
        line['docs']: List[Doc] = [doc | retriever.id_to_text(doc['doc_id']) for doc in docs]
        new_dataset.append(line)
    
    with open(output_file, 'w') as f:
        json.dump(new_dataset, f)
    del retriever
    return new_dataset
    

def compute_perplexity(teacher_path: str,
                       dataset: str | List[DataQA],
                       output_file: str,
                       image_folder: str,
                       num_chunks: int = 1,
                       chunk_idx: int = 0,
                       conv_mode: str = 'llava_v1'
                       )-> List[DataQA]:
    
    # Load dataset
    if isinstance(dataset, str):
        with open(dataset, 'r') as f:
            dataset = json.load(f)
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

    for line in tqdm(dataset):
        question, docs, responses = line['question'], line['docs'], line['responses']
        if 'image_file' in line['question'] and line['question']['image_file'] is not None:
            image_path = os.path.join(image_folder, line['question']['image_file'])
            image = Image.open(image_path)
        else:
            image = image_path = None
        is_vqa = image is not None


        ### 1. If the question is vqa, compute sentence-level ppl for each response ###
        if is_vqa:
            ppls_image = []
            num_sentences = {}
            context_prompts, full_prompts, image_paths = [], [], [] # num_responses * mean_num_sentences
            for i, response in enumerate(responses):
                sentences = split_in_sentences(response['text'])
                num_sentences[i] = len(sentences)
                for sentence in sentences:
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question['text'])
                    context_prompts.append(conv.get_prompt())
                    conv.append_message(conv.roles[1], sentence)
                    full_prompts.append(conv.get_prompt())
                    image_paths.append(image_path)
            res = teacher.compute_visual_conditional(full_prompts,
                                                     context_prompts,
                                                     image_paths)
            j = 0
            for i in range(len(responses)):
                input_lengths = res['input_lengths'][j:j+num_sentences[i]]
                scores = res['perplexities'][j:j+num_sentences[i]]
                exponents = [input_length / sum(input_lengths) for input_length in input_lengths]
                scores_scaled = [scores[i] ** exponents[i] for i in range(len(scores))]
                ppl_image = functools.reduce(lambda x, y: x*y, scores_scaled) # independence assumption
                ppls_image.append(ppl_image)
                j += num_sentences[i]


        ### 2. Compute retrieval-augmented ppl for each response ###
        doc_texts: List[str] = [doc['text'] for doc in docs]
        doc_concat: str = '\n'.join(doc_texts)
        question_doc: str = pad_image_token(f"Context:\n{doc_concat}\n\nQuestion: {question['text']}", 
                                            teacher.model.config.mm_use_im_start_end)
        context_prompts, full_prompts, image_paths = [], [], []
        for response in responses:
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question_doc)
            context_prompts.append(conv.get_prompt())
            conv.append_message(conv.roles[1], response['text'])
            full_prompts.append(conv.get_prompt())
            image_paths.append(image_path)
        if is_vqa:
            ppls_text = teacher.compute_visual_conditional(full_prompts,
                                                           context_prompts,
                                                           image_paths)['perplexities']
        else:
            # TODO: Not implemented. Conditionals.
            ppls_text = teacher.compute(full_prompts)['perplexities']


        ### 3. Merge ###
        ppls = ppls_text.copy()
        if is_vqa:
            ppls = [x+y for (x,y) in zip(ppls, ppls_image)]

        for i in range(len(responses)):
            line['responses'][i]['reward'] = -ppls[i]
            line['responses'][i]['reward_text'] = -ppls_text[i]
            if is_vqa:
                line['responses'][i]['reward_image'] = -ppls_image[i]

        for suffix in ["", "_text", "_image"]:
            if not is_vqa and suffix == "_image":
                continue
            line['top_response'+suffix] = max(line['responses'], 
                                              key=lambda r: r['reward'+suffix])
        new_dataset.append(line)

    
    with open(output_file, 'w') as f:
        json.dump(new_dataset, f)
    del teacher
    return new_dataset


def make_sft_dataset(dataset: str | List[DataQA],
                     image_folder: str,
                     output_file: str):
    """Creates an SFT dataset that can be directly fed to LLaVA for fine-tuning.
    Follows the format of: 
    https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
    """
    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if isinstance(dataset, str):
        with open(dataset, 'r') as f:
            dataset = json.load(f)

    for suffix in ["", "_text", "_image"]:
        sft_dataset = []

        for line in dataset:
            is_vqa = 'image_file' in line['question']
            if not is_vqa and suffix == '_image':
                continue

            finetune_data: LLaVAFinetuneData = {
                'id': line['top_response'+suffix]['rid'],
                'conversations': [
                    {
                        'from': 'human',
                        'value': pad_image_token(line['question']['text'], False) # TODO: mm start end might be true
                    },
                    {
                        'from': 'gpt',
                        'value': line['top_response'+suffix]['text']
                    }
                ]
            }

            if is_vqa:
                finetune_data['image'] = line['question']['image_file']
            sft_dataset.append(finetune_data)

        path = Path(output_file)
        output_path = str(path.with_name(path.stem + suffix + path.suffix))
        with open(output_path, 'w') as f:
            json.dump(sft_dataset, f)


def main(args) -> List[DataQA]:
    dataset: str = os.path.join(args.dataset_folder, args.dataset_name, f"{args.split}.json")
    if not os.path.isfile(dataset):
        raise FileNotFoundError
    
    ### Rollout ###
    print("Rollout start!")
    output_file = os.path.join(args.output_folder, args.dataset_name, f"rollout/{args.split}.json")
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            dataset: List[DataQA] = json.load(f)
        print("Output file found, proceeding to the next step...")
    else:
        dataset: List[DataQA] = rollout_model(student_path=args.rollout_student_path,
                                            student_base=args.rollout_student_base,
                                            dataset=dataset,
                                            output_file=output_file,
                                            image_folder=args.image_folder,
                                            num_return_sequences=args.rollout_num_return_sequences,
                                            max_new_tokens=args.rollout_max_new_tokens,
                                            conv_mode=args.conv_mode,
                                            num_chunks=args.num_chunks,
                                            chunk_idx=args.chunk_idx,
                                            batch_size=args.rollout_batch_size)
        print("Rollout complete!")

    ### Retrieval
    print("Retrieval start!")
    output_file = os.path.join(args.output_folder, args.dataset_name, f"retrieval/{args.split}.json")
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            dataset: List[DataQA] = json.load(f)
        print("Output file found, proceeding to the next step...")
    else:
        dataset: List[DataQA] = retrieve_docs(passages=args.retrieval_passages,
                                            passages_embeddings=args.retrieval_passages_embeddings,
                                            retriever_path=args.retrieval_retriever_path,
                                            dataset=dataset,
                                            output_file=output_file,
                                            n_docs=args.retrieval_n_docs
                                            )
        print("Retrieval complete!")

    ### Perplexity ###
    print("Reward computation start!")
    output_file = os.path.join(args.output_folder, args.dataset_name, f"reward/{args.split}.json")
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            dataset: List[DataQA] = json.load(f)
        print("Output file found, proceeding to the next step...")
    else:
        dataset: List[DataQA] = compute_perplexity(teacher_path=args.reward_teacher_path,
                                                dataset=dataset,
                                                output_file=output_file,
                                                image_folder=args.image_folder,
                                                conv_mode=args.conv_mode)
        print("Reward computation complete!")

    ### SFT ###
    dataset: List[DataQA] = make_sft_dataset(dataset=dataset,
                                             image_folder=args.image_folder,
                                             output_file=os.path.join(args.output_folder, 
                                                                      args.dataset_name, 
                                                                      f"{args.split}.json"))
    print("You are ready for SFT!")
    return dataset
                                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dataset-folder", type=str, default=os.path.join(BASE_DIR, "data_in/iter0")) # SUBJECT TO CHANGE
    parser.add_argument("--image-folder", type=str, default=os.path.join(BASE_DIR, "data"))
    parser.add_argument("--output-folder", type=str, default=os.path.join(BASE_DIR, "data_in/iter1")) # SUBJECT TO CHANGE
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default='llava_v1')

    ### Rollout ###
    parser.add_argument("--rollout-student-path", type=str, 
                        default=os.path.join(BASE_DIR, "models/llava-v1.5-7b"))
    parser.add_argument("--rollout-student-base", type=str, default=None)
    parser.add_argument("--rollout-num-return-sequences", type=int, default=3)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=128)
    parser.add_argument("--rollout-batch-size", type=int, default=8)

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
    # parser.add_argument("--reward-lambda-const", type=float, default=0.4)
    args = parser.parse_args()

    if args.dataset_name is None:
        # for dataset_name in ['aokvqa', 's3vqa', 'scienceqa', 'okvqa']:
        for dataset_name in ['aokvqa', 'okvqa']:
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
