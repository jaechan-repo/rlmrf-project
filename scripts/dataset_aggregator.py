import sys
sys.path.append("../")

import pandas as pd
import os
import json
from collections import Counter
from rampa.mytyping import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', type=str, 
                        default= "/mmfs1/gscratch/ark/chan0369/rampa-project/data",
                        help="path of the root folder that contains the following datasets: aokvqa, s3vqa, okvqa, scienceqa")
    parser.add_argument('-o', '--out_path', type=str,
                        default="/mmfs1/gscratch/ark/chan0369/rampa-project/data_prep/iter0",
                        help="path of the output folder")
    args = parser.parse_args()
    in_path, out_path = args.in_path, args.out_path
    datasets = ['aokvqa', 's3vqa', 'okvqa', 'scienceqa']
    splits = ['train', 'val']

    for dataset in datasets:
        out_data_path = os.path.join(out_path, dataset)
        if not os.path.exists(out_data_path):
            os.mkdir(out_data_path)
        for split in splits:
            out = []
            if dataset == 'aokvqa':
                in_file = os.path.join(in_path, dataset, f"aokvqa_v1p0_{split}.json")
                df = pd.read_json(in_file)
                df = df[~df['difficult_direct_answer']].reset_index(drop=True)
                for _, row in df.iterrows():
                    line: DataQA = {
                        'question': {
                            'qid': row['question_id'], 
                            'text': row['question'], 
                            'image_id': row['image_id'],
                            'image_file': f"coco/{split}2017/{str(row['image_id']).zfill(12)}.jpg",
                        },
                        'ref_ans': {
                            'direct_answers': row['direct_answers'],
                            'text': row['choices'][row['correct_choice_idx']],
                            'rationales': row['rationales']
                        },
                        'split': split,
                        'dataset': dataset
                        # 'misc': {
                        #     'dfficult_direct_answer': row['difficult_direct_answer']
                        # },
                    }
                    out.append(line)
            elif dataset == 's3vqa':
                df_questions = pd.read_json(os.path.join(in_path, dataset, f"S3-VQA_{split if split != 'val' else 'dev'}_questions.json"))
                df_annotations = pd.read_json(os.path.join(in_path, dataset, f"S3-VQA_{split if split != 'val' else 'dev'}_annotations.json"))
                df = df_questions.merge(df_annotations, how='inner')
                for _, row in df.iterrows():
                    line: DataQA = {
                        'question': {
                            'qid': row['question_id'], 
                            'text': row['question'], 
                            'image_id': row['image_id'],
                            'image_file': f"openimages/s3vqa/{row['image_id']}.jpg",
                        },
                        'ref_ans': {
                            'text': row['answer']['raw'],
                            'stem': row['answer']['answer'],
                            'hyponym': row['hyponym'],
                            'hypernym': row['hypernym']
                        },
                        'split': split,
                        'dataset': dataset
                    }
                    out.append(line)
            elif dataset == 'okvqa':
                with open(os.path.join(in_path, dataset, f"OpenEnded_mscoco_{split}2014_questions.json")) as f:
                    df_questions = pd.DataFrame(json.load(f)['questions'])
                with open(os.path.join(in_path, dataset, f"mscoco_{split}2014_annotations.json")) as f:
                    df_annotations = pd.DataFrame(json.load(f)['annotations'])
                df = df_annotations.merge(df_questions, how='inner')
                df = df[df['confidence'] == 5].reset_index(drop=True)
                for _, row in df.iterrows():
                    line: DataQA = {
                        'question': {
                            'qid': row['question_id'], 
                            'text': row['question'], 
                            'image_id': row['image_id'],
                            'image_file': f"coco/{split}2017/{str(row['image_id']).zfill(12)}.jpg",
                        },
                        'ref_ans': {
                            'text': Counter([a['raw_answer'] for a in row['answers']]).most_common(1)[0][0],
                            'stem': Counter([a['answer'] for a in row['answers']]).most_common(1)[0][0]
                        },
                        'misc': {
                            'confidence': row['confidence'],
                            'question_type': row['question_type'],
                            'answer_type': row['answer_type'],
                        },
                        'split': split,
                        'dataset': dataset
                    }
                    out.append(line)
            elif dataset == 'scienceqa':
                with open(os.path.join(in_path, dataset, "problems.json")) as f:
                    df = pd.DataFrame(json.load(f)).T
                    df = df[df['split']==split] # do not reset index!
                choice_prefixes = [chr(ord('A') + i) for i in range(26)] # A-Z
                def format_options(options, choice_prefixes):
                    return ' '.join([f'({c}) {o}' for c, o in zip(choice_prefixes, options)])
                def format_prompt(r, choice_prefixes):
                    options = format_options(r['choices'], choice_prefixes)
                    # context = f"Context: {r['hint']}\n" if r['hint'].strip() != "" else ""
                    # return f'''{context}Question: {r["question"]}\nOptions:{options}'''
                    return f'''{r["question"]}\nOptions: {options}'''
                def format_label(r, choice_prefixes):
                    # letter_answer, direct_answer
                    return choice_prefixes[r['answer']], r['choices'][r['answer']]
                for i, row in df.iterrows():
                    question_with_choices = format_prompt(row, choice_prefixes=choice_prefixes)
                    letter_answer, direct_answer = format_label(row, choice_prefixes=choice_prefixes)
                    line: DataQA = {
                        'question': {
                            'qid': i,
                            'without_choices': row['question'],
                            'choices': row['choices'],
                            'text': question_with_choices
                        },
                        'ref_ans': {
                            'index': row['answer'],
                            'letter_answer': letter_answer,
                            'direct_answer': direct_answer,
                            'text': f"{letter_answer} {direct_answer}"
                        },
                        'misc': {
                            'hint': row['hint'],
                            'task': row['task'],
                            'solution': row['solution'],
                        },
                        'split': split,
                        'dataset': dataset
                    }
                    if row['image'] is not None:
                        line['question']['image_file'] = f"scienceqa/{split}/{i}"
                    out.append(line)
            else:
                raise ValueError("dataset name not featured")
            out_file = os.path.join(out_data_path, f"{split}.json")
            with open(out_file, 'w') as f:
                json.dump(out, f)
        print(f"{dataset} conversion finished!")